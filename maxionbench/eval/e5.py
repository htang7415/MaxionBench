"""E5 quality and cost: one model on RAG (CRAG, HotpotQA), BFCL, and agentic HotpotQA over MCP.

Answer requests run one at a time (concurrency 1) so latencies are comparable between Gemini and a
local engine; judge requests run in parallel. Items are split into seeded shards that serve as the
result schema's repeats, so each cell's CI reflects item sampling. RAG correctness comes from the
Gemini judge (rubric qa-judge-v1); EM/F1 are reported alongside.

    python -m maxionbench.eval.e5 experiments/e5_gemini.yaml [--out artifacts/e5] [--limit 10]
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Callable

import yaml

from maxionbench.agents.hotpot_env import DEFAULT_DATASET, HotpotCorpus, build_tasks
from maxionbench.agents.loop import ChatPolicy, run_tasks
from maxionbench.datasets.loaders.v03 import BFCL_CATEGORIES, load_bfcl, load_crag
from maxionbench.eval.batch import Call, bound_send, metered, retryable, run_calls
from maxionbench.eval.qa import QAItem, crag_item, hotpot_items
from maxionbench.eval.tool_client import chat_tools
from maxionbench.graders import bfcl
from maxionbench.graders.judge import RUBRIC_VERSION, judge_messages, parse_label
from maxionbench.graders.qa import agent_success, crag_score, grade_qa
from maxionbench.harness.budget import ModelPrice, cost_usd
from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, Provenance, TrialResult, aggregate_cells
from maxionbench.harness.runner import _git, _scrubber
from maxionbench.harness.targets import GeminiTarget, Target, make_target
from maxionbench.metrics.latency import latency_summary
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.schemas.result_schema import stable_config_fingerprint, utc_now_iso

E5_SCHEMA = "maxionbench-e5-v1"
SUITES = ("rag_crag", "rag_hotpot", "bfcl", "agent")


@dataclass
class ItemResult:
    suite: str
    id: str
    status: str
    correct: bool
    latency_s: float | None
    ttft_s: float | None
    cost_usd: float
    detail: dict[str, Any] = field(default_factory=dict)


def load_e5_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != E5_SCHEMA:
        raise ValueError(f"schema_version must be {E5_SCHEMA!r}")
    unknown = set(spec["suites"]) - set(SUITES)
    if unknown:
        raise ValueError(f"unknown suites {sorted(unknown)}")
    return spec


def _item_cost(price: ModelPrice | None, results: list[Any]) -> float:
    if price is None:
        return 0.0
    return sum(cost_usd(price, input_tokens=r.prompt_tokens, output_tokens=r.completion_tokens,
                        cached_tokens=r.cached_tokens) for r in results)


def run_rag(target: Target, items: list[QAItem], judge: Target, label: str, log: Callable[[str], None]
            ) -> tuple[list[ItemResult], dict[str, Any]]:
    price = _price(target)
    answers, spend = run_calls(target, [Call(i.id, i.messages, max_tokens=64) for i in items], label, workers=1)
    log(f"{label}: {len(items)} answers, ${spend['spend_usd']:.4f}")
    judge_calls = [Call(i.id, judge_messages(i.question, i.golds, answers[i.id].text.strip()), max_tokens=512)
                   for i in items if answers[i.id].status == "ok"]
    verdicts, judge_spend = run_calls(judge, judge_calls, f"{label}/judge", workers=8)
    out = []
    for item in items:
        r = answers[item.id]
        answer = r.text.strip()
        judged = parse_label(verdicts[item.id].text) if item.id in verdicts else None
        qa = grade_qa(answer, item.golds)
        out.append(ItemResult(
            item.source, item.id, r.status, judged == "correct", r.e2e_s if r.status == "ok" else None, r.ttft_s,
            _item_cost(price, [r]),
            {"answer": answer, "judge_label": judged or "unjudged", "em": qa.em, "f1": qa.f1, "error": r.error},
        ))
    return out, {"answer": spend, "judge": judge_spend}


def run_bfcl(target: Target, per_category: int, seed: int, label: str, log: Callable[[str], None]
             ) -> tuple[list[ItemResult], dict[str, Any]]:
    rng = random.Random(seed)
    cases = [c for cat in BFCL_CATEGORIES for c in rng.sample(load_bfcl(cat), min(per_category, len(load_bfcl(cat))))]
    calls = [Call(c.id, c.messages, max_tokens=512, tools=tuple(bfcl.to_openai_tools(c.functions))) for c in cases]
    results, spend = run_calls(target, calls, label, workers=1, send=chat_tools)
    log(f"{label}: {len(cases)} cases, ${spend['spend_usd']:.4f}")
    price = _price(target)
    out = []
    for case in cases:
        r = results[case.id]
        grade, calls_made = bfcl.BfclGrade(False, r.error), []
        if r.status == "ok":
            message = json.loads(r.text)
            try:
                calls_made = bfcl.parse_openai_tool_calls(message.get("tool_calls"))
                grade = bfcl.grade(case, calls_made) if calls_made else bfcl.BfclGrade(False, "no tool call")
            except ValueError as exc:
                grade = bfcl.BfclGrade(False, str(exc))
        out.append(ItemResult("bfcl", case.id, r.status, grade.correct, r.e2e_s if r.status == "ok" else None, None,
                              _item_cost(price, [r]),
                              {"category": case.category, "calls": calls_made, "grade_error": grade.error}))
    return out, {"answer": spend}


def run_agent_suite(target: Target, n: int, max_steps: int, seed: int, label: str, log: Callable[[str], None]
                    ) -> tuple[list[ItemResult], dict[str, Any]]:
    corpus = HotpotCorpus(DEFAULT_DATASET)
    tasks = build_tasks(corpus, n, seed)
    del corpus  # the MCP server process builds its own index
    fn = bound_send(target, chat_tools)
    max_tokens = 256

    def estimate(price: ModelPrice) -> float:  # every step at a generous context size
        return len(tasks) * max_steps * cost_usd(price, input_tokens=4000, output_tokens=max_tokens)

    policies: dict[str, ChatPolicy] = {}
    with metered(target, estimate, label) as meter:
        def send(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
            for attempt in range(3):
                r = fn(target.base_urls[0], messages, tools=tools, max_tokens=max_tokens, timeout_s=60.0)
                meter.add(r, messages, max_tokens)
                if r.status == "ok" or not retryable(r.error):
                    return r
                time.sleep(2.0 * (attempt + 1))
            return r

        def policy_for(task: Any) -> ChatPolicy:
            policies[task.task_id] = ChatPolicy(send)
            return policies[task.task_id]

        runs = run_tasks(tasks, policy_for, DEFAULT_DATASET, max_steps=max_steps)
    log(f"{label}: {len(tasks)} tasks, ${meter.spend_usd:.4f}")
    price = _price(target)
    out = []
    for task, run in zip(tasks, runs):
        steps = policies[task.task_id].results
        out.append(ItemResult(
            "agent", task.task_id, "ok" if run.status != "error" else "error", agent_success(run.answer, task.answer),
            sum(r.e2e_s for r in steps) if run.status != "error" else None, None, _item_cost(price, steps),
            {"answer": run.answer, "gold": task.answer, "run_status": run.status, "model_calls": len(steps),
             "tool_calls": [c["name"] for c in run.tool_calls], "error": run.error},
        ))
    return out, {"answer": {**meter.usage, "spend_usd": round(meter.spend_usd, 6)}}


def _price(target: Target) -> ModelPrice | None:
    pricing = target.pricing()
    return None if pricing is None else pricing[1]


def shard_metrics(items: list[ItemResult]) -> dict[str, float]:
    ok = [i for i in items if i.status == "ok"]
    correct = sum(i.correct for i in items)
    spend = sum(i.cost_usd for i in items)
    m: dict[str, float] = {
        "items": float(len(items)),
        "errors": float(len(items) - len(ok)),
        "accuracy": correct / len(items),
        "spend_usd": spend,
        "usd_per_1k_items": 1000 * spend / len(items),
    }
    if correct:
        m["usd_per_correct"] = spend / correct
    latencies = [i.latency_s * 1000 for i in ok if i.latency_s is not None]
    if latencies:
        m.update({f"latency_{k}": v for k, v in latency_summary(latencies).items()})
    ttfts = [i.ttft_s * 1000 for i in ok if i.ttft_s is not None]
    if ttfts:
        m.update({f"ttft_{k}": v for k, v in latency_summary(ttfts).items()})
    if items[0].suite in ("crag", "hotpotqa"):
        m["em"] = sum(i.detail["em"] for i in items) / len(items)
        m["f1"] = sum(i.detail["f1"] for i in items) / len(items)
        labels = [i.detail["judge_label"] for i in items if i.detail["judge_label"] in ("correct", "incorrect", "missing")]
        m["missing_rate"] = sum(label == "missing" for label in labels) / len(items)
        if items[0].suite == "crag" and labels:
            m["crag_score"] = crag_score(labels)
    if items[0].suite == "agent":
        m["model_calls_mean"] = sum(i.detail["model_calls"] for i in items) / len(items)
        m["tool_calls_mean"] = sum(len(i.detail["tool_calls"]) for i in items) / len(items)
        m["answered_rate"] = sum(i.detail["run_status"] == "answered" for i in items) / len(items)
    return m


def _capped(n: int, limit: int | None) -> int:
    return n if limit is None else min(n, limit)


def run_e5(spec: dict[str, Any], out_root: Path, limit: int | None = None,
           log: Callable[[str], None] = lambda m: print(f"[e5] {m}", file=sys.stderr, flush=True)) -> Path:
    started_at = utc_now_iso()
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    seed, shards, suites = int(spec.get("seed", 0)), int(spec.get("shards", 5)), spec["suites"]
    model = spec["model"]
    judge = GeminiTarget({"model": spec["judge"]["model"], "reasoning_effort": spec["judge"].get("reasoning_effort")})
    items: dict[str, list[ItemResult]] = {}
    spends: dict[str, Any] = {}
    target = make_target(model["kind"], dict(model.get("params") or {}), out_dir / "logs")
    with target:
        label = f"e5/{spec['name']}"
        if "rag_crag" in suites:
            rng = random.Random(seed)
            examples = rng.sample(load_crag(), _capped(int(suites["rag_crag"]["n"]), limit))
            items["rag_crag"], spends["rag_crag"] = run_rag(target, [crag_item(e) for e in examples], judge,
                                                            f"{label}/rag_crag", log)
        if "rag_hotpot" in suites:
            qa = hotpot_items(DEFAULT_DATASET, _capped(int(suites["rag_hotpot"]["n"]), limit), seed)
            items["rag_hotpot"], spends["rag_hotpot"] = run_rag(target, qa, judge, f"{label}/rag_hotpot", log)
        if "bfcl" in suites:
            items["bfcl"], spends["bfcl"] = run_bfcl(target, _capped(int(suites["bfcl"]["per_category"]), limit), seed,
                                                     f"{label}/bfcl", log)
        if "agent" in suites:
            items["agent"], spends["agent"] = run_agent_suite(
                target, _capped(int(suites["agent"]["n"]), limit), int(suites["agent"].get("max_steps", 8)), seed,
                f"{label}/agent", log)
        target_desc = target.describe()

    scrub, key_present = _scrubber()
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for suite_items in items.values():
            for item in suite_items:
                fh.write(scrub(json.dumps(asdict(item), ensure_ascii=False)) + "\n")
    trials, cell_params = [], {}
    for suite, suite_items in items.items():
        cell_params[suite] = {"suite": suite, "model": model}
        for s in range(shards):
            shard = suite_items[s::shards]
            if not shard:
                continue
            trials.append(TrialResult(
                trial_id=f"{suite}-shard{s}", cell_id=suite, repeat=s, seed=seed, status="ok", started_at=started_at,
                duration_s=0.0, host_load_1m_before=0.0, quiet_host_ok=True, metrics=shard_metrics(shard),
                requests_per_endpoint=[len(shard)], target={**target_desc, "spend": spends[suite]}, error=None))
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"],
        description=str(spec.get("description", "")), spec=spec,
        provenance=Provenance(
            git_commit=_git(["rev-parse", "HEAD"]) or "unknown", git_dirty=bool(_git(["status", "--porcelain"])),
            spec_fingerprint=stable_config_fingerprint(spec), started_at=started_at, finished_at=utc_now_iso(),
            host=collect_system_info(),
            tools={"python": platform.python_version(), "harness_result_schema": RESULT_SCHEMA_VERSION,
                   "gemini_key_present": key_present, "judge_rubric": RUBRIC_VERSION, "item_limit": limit,
                   "trials_planned": len(trials), "trials_completed": len(trials)}),
        trials=trials, cells=aggregate_cells(trials, cell_params))
    (out_dir / "results.json").write_text(scrub(json.dumps(result.to_dict(), indent=2)) + "\n", encoding="utf-8")
    overall = {suite: shard_metrics(v) for suite, v in items.items()}
    (out_dir / "summary.json").write_text(json.dumps({"overall": overall, "spend": spends}, indent=2) + "\n",
                                          encoding="utf-8")
    log(f"wrote {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="E5 quality and cost")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/e5"))
    parser.add_argument("--limit", type=int, help="cap items per suite (pilot runs)")
    args = parser.parse_args(argv)
    out_dir = run_e5(load_e5_spec(args.spec), args.out, args.limit)
    print(json.dumps(json.loads((out_dir / "summary.json").read_text())["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
