"""Step 3: context policies on agentic BrowseComp-Plus with Gemini: accuracy, billed cost, and cache hits.

Every policy runs every task (task-major order, so a budget stop leaves complete, paired task groups).
Each run's system prompt starts with a unique run id, so Gemini's implicit cache cannot share prefixes
between runs (policies on one task would otherwise reuse each other's cached prompts). Cost is what
Gemini bills: input at the cached or uncached rate as reported, plus output and reasoning tokens, plus
summarizer calls. Correctness comes from the calibrated judge. Outputs hold ids and numbers only.

    python -m maxionbench.eval.context_eval experiments/c1_context_policies.yaml [--limit 3]
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence
import uuid

import yaml

from maxionbench.agents.browsecomp_env import SYSTEM_PROMPT, BrowseTask, load_tasks
from maxionbench.agents.context import Message, estimate_tokens, make_policy
from maxionbench.agents.loop import ChatPolicy
from maxionbench.eval.agent_trial import JUDGE, MAX_TOKENS, run_task
from maxionbench.eval.batch import Call, Meter, bound_send, metered, retryable, run_calls
from maxionbench.eval.tool_client import chat_tools
from maxionbench.graders.judge import RUBRIC_VERSION, judge_messages, parse_label
from maxionbench.harness.budget import ModelPrice, cost_usd
from maxionbench.harness.provenance import make_provenance, scrubber
from maxionbench.harness.results import (
    RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells, mean_ci,
)
from maxionbench.harness.targets import GeminiTarget
from maxionbench.rag.llm_client import chat_completion
from maxionbench.schemas.result_schema import utc_now_iso

SCHEMA = "maxionbench-context-v1"
SUMMARY_MAX_TOKENS = 1024
SUMMARIZE_PROMPT = (
    "You are condensing an agent's earlier work on a research question so it can continue with less context. "
    "Write a compact summary: the clues of the question, what each search found (page ids and key facts), "
    "candidate answers with their evidence, and what remains to check. Do not answer the question yourself."
)


def _log(msg: str) -> None:
    print(f"[context-eval] {msg}", file=sys.stderr, flush=True)


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA!r}")
    if "full" not in spec["policies"]:
        raise ValueError("policies must include the `full` baseline")
    return spec


def transcript(messages: Sequence[Message]) -> str:
    """Plain-text rendering for the summarizer (no tool schema needed)."""
    lines = []
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or ():
                lines.append(f"[agent called {c['function']['name']}({c['function'].get('arguments')})]")
            if m.get("content"):
                lines.append(f"[agent] {m['content']}")
        elif m["role"] == "tool":
            lines.append(f"[tool result] {m.get('content')}")
        else:
            lines.append(f"[{m['role']}] {m.get('content')}")
    return "\n".join(lines)


class Recorder:
    """Wraps a context policy and records history vs view size per step."""

    def __init__(self, policy: Any) -> None:
        self.policy, self.name = policy, policy.name
        self.steps: list[tuple[int, int]] = []

    def view(self, history: Sequence[Message]) -> list[Message]:
        v = self.policy.view(history)
        self.steps.append((estimate_tokens(history), estimate_tokens(v)))
        return v

    def reset(self) -> None:
        self.policy.reset()


def _call_cost(price: ModelPrice, results: Sequence[Any]) -> float:
    return sum(cost_usd(price, input_tokens=r.prompt_tokens, output_tokens=r.completion_tokens + r.reasoning_tokens,
                        cached_tokens=r.cached_tokens) for r in results if r.status == "ok")


def run_context_eval(spec: dict[str, Any], out_root: Path, limit: int | None = None) -> Path:
    started_at = utc_now_iso()
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    n = int(spec["tasks"]) if limit is None else min(int(spec["tasks"]), limit)
    seed, pool, max_steps, budget = int(spec["seed"]), int(spec["pool"]), int(spec["max_steps"]), float(spec["budget_usd"])
    policies: dict[str, dict[str, Any]] = {name: dict(params or {}) for name, params in spec["policies"].items()}
    tasks = load_tasks(n, seed, pool=pool)
    target, judge = GeminiTarget(spec["model"]), GeminiTarget(JUDGE)
    price = target.pricing()[1]
    send_tools, send_text = bound_send(target, chat_tools), bound_send(target, chat_completion)

    def estimate(p: ModelPrice) -> float:  # the budget plus one task group at 60k uncached tokens per step
        return budget + len(policies) * max_steps * cost_usd(p, input_tokens=60_000, output_tokens=MAX_TOKENS)

    def with_retries(fn: Any, meter: Meter, messages: Any, max_tokens: int, **kwargs: Any) -> Any:
        for attempt in range(3):
            r = fn(target.base_urls[0], messages, max_tokens=max_tokens, timeout_s=120.0, **kwargs)
            meter.add(r, messages, max_tokens)
            if r.status == "ok" or not retryable(r.error):
                return r
            time.sleep(2.0 * (attempt + 1))
        return r

    items: list[dict[str, Any]] = []
    with target, metered(target, estimate, f"context-eval/{spec['name']}") as meter, \
            tempfile.TemporaryDirectory() as tmp:
        for i, task in enumerate(tasks):
            if meter.spend_usd >= budget:
                _log(f"budget ${budget:.2f} reached after {i} tasks")
                break
            for name, params in policies.items():
                items.append(run_one(task, name, params, max_steps, Path(tmp), meter, price, with_retries,
                                     send_tools, send_text))
            _log(f"task {i + 1}/{len(tasks)} done, spend ${meter.spend_usd:.4f}")
    agent_spend = meter.spend_usd

    calls = [Call(f"{it['task_id']}|{it['policy']}", judge_messages(it["_question"], [it["_gold"]], it["_text"]),
                  max_tokens=512) for it in items if it["_answer"]]
    with judge:
        verdicts, judge_usage = run_calls(judge, calls, f"context-eval/{spec['name']}/judge", workers=8) \
            if calls else ({}, {"spend_usd": 0.0})
    for it in items:  # drop task text before anything is written
        for k in ("_question", "_gold", "_text", "_answer"):
            del it[k]
        verdict = verdicts.get(f"{it['task_id']}|{it['policy']}")
        it["judge_label"] = (parse_label(verdict.text) if verdict else None) or "unjudged"
        it["correct"] = it["judge_label"] == "correct"

    scrub, key_present = scrubber()
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(scrub(json.dumps(it)) + "\n")
    task_ids = list(dict.fromkeys(it["task_id"] for it in items))
    shards = max(1, min(int(spec.get("shards", 5)), len(task_ids)))
    trials = []
    for name in policies:
        for s in range(shards):
            shard_ids = set(task_ids[s::shards])
            shard = [it for it in items if it["policy"] == name and it["task_id"] in shard_ids]
            trials.append(TrialResult(
                trial_id=f"{name}-shard{s}", cell_id=name, repeat=s, seed=seed, status="ok", started_at=started_at,
                duration_s=0.0, host_load_1m_before=0.0, quiet_host_ok=True, metrics=metrics(shard),
                requests_per_endpoint=[sum(it["model_calls"] for it in shard)], target=target.describe(), error=None))
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"],
        description=str(spec.get("description", "")), spec=spec,
        provenance=make_provenance(spec, started_at, {
            "gemini_key_present": key_present, "judge_rubric": RUBRIC_VERSION, "task_limit": limit,
            "tasks_completed": len(task_ids), "spend_usd": {"agent": round(agent_spend, 4),
                                                             "judge": judge_usage["spend_usd"]}}),
        trials=trials, cells=aggregate_cells(trials, {name: {"policy": name, **p} for name, p in policies.items()}))
    (out_dir / "results.json").write_text(scrub(json.dumps(result.to_dict(), indent=2)) + "\n", encoding="utf-8")
    summary = {"tasks": len(task_ids), "spend_usd": {"agent": round(agent_spend, 4), "judge": judge_usage["spend_usd"]},
               "overall": {name: metrics([it for it in items if it["policy"] == name]) for name in policies},
               "paired_vs_full": paired_vs_full(items, list(policies))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote {out_dir}")
    return out_dir


def run_one(task: BrowseTask, name: str, params: dict[str, Any], max_steps: int, workdir: Path, meter: Meter,
            price: ModelPrice, with_retries: Any, send_tools: Any, send_text: Any) -> dict[str, Any]:
    summaries: list[Any] = []

    def summarizer(messages: Sequence[Message]) -> str:
        prompt = [{"role": "system", "content": SUMMARIZE_PROMPT}, {"role": "user", "content": transcript(messages)}]
        r = with_retries(send_text, meter, prompt, SUMMARY_MAX_TOKENS)
        summaries.append(r)
        return r.text.strip() if r.status == "ok" else "(summary unavailable)"

    context = Recorder(make_policy(name, summarizer if name.startswith("summarize") else None, **params))
    agent = ChatPolicy(lambda messages, tools: with_retries(send_tools, meter, messages, MAX_TOKENS, tools=tools))
    nonce = f"run {uuid.uuid4().hex}\n"  # unique first tokens: no implicit-cache sharing across runs
    run = run_task(task, agent, max_steps, workdir, context, nonce + SYSTEM_PROMPT)
    steps = [r for r in agent.results if r.status == "ok"]
    prompt = sum(r.prompt_tokens for r in steps)
    history_tokens = sum(h for h, _ in context.steps)
    return {
        "task_id": task.task_id, "policy": name, "run_status": run.status,
        "error_type": (run.error or "").split(":", 1)[0] or None,  # type only: messages may quote task text
        "model_calls": len(agent.results), "summary_calls": len(summaries),
        "tool_calls": len(run.tool_calls),
        "prompt_tokens": prompt, "cached_tokens": sum(r.cached_tokens for r in steps),
        "output_tokens": sum(r.completion_tokens + r.reasoning_tokens for r in steps),
        "peak_context_tokens": max((r.prompt_tokens for r in steps), default=0),
        "view_share": round(sum(v for _, v in context.steps) / history_tokens, 4) if history_tokens else 1.0,
        "edits": getattr(context.policy, "edits", getattr(context.policy, "compactions", 0)),  # rewrites of the view
        "cost_usd": round(_call_cost(price, agent.results) + _call_cost(price, summaries), 6),
        "summary_cost_usd": round(_call_cost(price, summaries), 6),
        "_answer": run.status == "answered" and bool(run.answer), "_text": run.answer or "",
        "_question": task.question, "_gold": task.answer,
    }


def metrics(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    n = len(items)
    correct = sum(it["correct"] for it in items)
    cost = sum(it["cost_usd"] for it in items)
    prompt = sum(it["prompt_tokens"] for it in items)
    m = {
        "tasks": float(n), "accuracy": correct / n, "cost_usd_per_task": cost / n,
        "cached_share": sum(it["cached_tokens"] for it in items) / prompt if prompt else 0.0,
        "prompt_tokens_per_task": prompt / n, "peak_context_tokens": sum(it["peak_context_tokens"] for it in items) / n,
        "model_calls_per_task": sum(it["model_calls"] for it in items) / n,
        "view_share": sum(it["view_share"] for it in items) / n,
        "summary_cost_share": sum(it["summary_cost_usd"] for it in items) / cost if cost else 0.0,
        "answered_rate": sum(it["run_status"] == "answered" for it in items) / n,
        "edits_per_task": sum(it.get("edits", 0) for it in items) / n,
    }
    if correct:
        m["usd_per_correct"] = cost / correct
    return m


def paired_vs_full(items: Sequence[dict[str, Any]], policies: Sequence[str]) -> dict[str, Any]:
    """Per-task differences against `full` on the same tasks (policy minus full), mean with 95% CI."""
    by = {(it["task_id"], it["policy"]): it for it in items}
    tasks = sorted({t for t, _ in by})
    out = {}
    for name in policies:
        if name == "full":
            continue
        pairs = [(by[(t, name)], by[(t, "full")]) for t in tasks if (t, name) in by and (t, "full") in by]
        if len(pairs) < 2:
            continue
        out[name] = {"tasks": len(pairs)}
        for field in ("correct", "cost_usd", "prompt_tokens"):
            ci = mean_ci([float(a[field]) - float(b[field]) for a, b in pairs])
            out[name][field] = {"mean": round(ci.mean, 6), "ci95": [round(ci.ci_low, 6), round(ci.ci_high, 6)]}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Context policies on BrowseComp-Plus (Gemini)")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/context_eval"))
    parser.add_argument("--limit", type=int, help="cap tasks (pilot)")
    parser.add_argument("--budget-usd", type=float, help="override the spec budget (pilot)")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    if args.budget_usd is not None:
        spec["budget_usd"] = args.budget_usd
    out_dir = run_context_eval(spec, args.out, args.limit)
    print(json.dumps(json.loads((out_dir / "summary.json").read_text())["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
