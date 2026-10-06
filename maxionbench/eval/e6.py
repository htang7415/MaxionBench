"""E6 caching economics on Gemini: implicit caching vs explicit context caching vs the Batch API.

One workload for every arm: sessions of several questions over one shared long document (HotpotQA
gold paragraphs for the session's questions plus distractors, >4,096 tokens, Gemini 3.x's caching
minimum). Every (arm, repeat) uses its own sessions, so no arm is warmed by another's cache.

- implicit: OpenAI-compatible streaming requests, one session at a time; caching is automatic.
- explicit: per session, the system prompt and document go into a `cachedContents` entry (TTL
  billed in full, a conservative storage charge); requests send only the question and reference it.
- batch: all requests in one inline `batchGenerateContent` job at the batch price; latency is the
  job's turnaround, not per-request.

    python -m maxionbench.eval.e6 [--sessions 8 --questions 5 --repeats 3] [--arms implicit,explicit,batch]
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import yaml

from maxionbench.agents.hotpot_env import DEFAULT_DATASET
from maxionbench.eval.batch import metered
from maxionbench.graders.qa import grade_qa
from maxionbench.harness.budget import ModelPrice, cost_usd
from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, Provenance, TrialResult, aggregate_cells
from maxionbench.harness.runner import _git, _scrubber
from maxionbench.harness.secrets import load_gemini_key, redact
from maxionbench.harness.targets import GeminiTarget
from maxionbench.metrics.latency import latency_summary
from maxionbench.rag.llm_client import CompletionResult, chat_completion
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.schemas.result_schema import stable_config_fingerprint, utc_now_iso
from maxionbench.tools.rag_eval import SYSTEM_PROMPT

MODEL = "gemini-3.5-flash-lite"
REST = "https://generativelanguage.googleapis.com/v1beta"
MAX_TOKENS = 64
CACHE_TTL_S = 300
ARMS = ("implicit", "explicit", "batch")


@dataclass(frozen=True)
class Session:
    id: str
    document: str
    questions: tuple[tuple[str, str, str], ...]  # (question_id, text, gold answer)


@dataclass(frozen=True)
class Answer:
    qid: str
    text: str
    status: str
    ttft_s: float | None
    e2e_s: float | None
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int


def build_sessions(n: int, questions: int, distractors: int, seed: int, dataset_dir: Path = DEFAULT_DATASET
                   ) -> list[Session]:
    rng = random.Random(seed)
    docs: dict[str, str] = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            docs[row["doc_id"]] = row["text"]
    gold: dict[str, list[str]] = {}
    with (dataset_dir / "qrels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            gold.setdefault(qid, []).append(doc_id)
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as fh:
        queries = [r for r in map(json.loads, fh) if r["query_id"] in gold]
    picked = rng.sample(queries, n * questions)
    doc_ids = list(docs)
    sessions = []
    for s in range(n):
        qs = picked[s * questions:(s + 1) * questions]
        ids = list(dict.fromkeys(d for q in qs for d in gold[q["query_id"]]))
        ids += [d for d in rng.sample(doc_ids, distractors) if d not in ids]
        rng.shuffle(ids)
        document = "\n\n".join(f"[{i}] {docs[d]}" for i, d in enumerate(ids, start=1))
        sessions.append(Session(f"seed{seed}-s{s}", document, tuple(
            (q["query_id"].rsplit("::", 1)[-1], q["text"], q["answer"]) for q in qs)))
    return sessions


def _rest(method: str, path: str, body: dict[str, Any] | None = None, timeout_s: float = 60.0) -> dict[str, Any]:
    secret = load_gemini_key()
    req = urllib.request.Request(f"{REST}/{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-goog-api-key": secret.reveal()})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace")
        raise RuntimeError(redact(f"{method} {path}: http {exc.code}: {detail}", secret)) from None
    return json.loads(raw) if raw else {}


def _answer(qid: str, r: CompletionResult) -> Answer:
    return Answer(qid, r.text.strip(), r.status, r.ttft_s, r.e2e_s if r.status == "ok" else None,
                  r.prompt_tokens, r.cached_tokens, r.completion_tokens)


def run_implicit(sessions: list[Session], send: Callable[..., CompletionResult], meter: Any) -> tuple[list[Answer], float]:
    out = []
    for s in sessions:
        for qid, text, _ in s.questions:
            messages = ({"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{s.document}\n\nQuestion: {text}"})
            r = send(GeminiTarget.BASE_URL, messages, max_tokens=MAX_TOKENS, timeout_s=60.0)
            meter.add(r, messages, MAX_TOKENS)
            out.append(_answer(qid, r))
    return out, 0.0


def run_explicit(sessions: list[Session], send: Callable[..., CompletionResult], meter: Any, price: ModelPrice
                 ) -> tuple[list[Answer], float]:
    """Returns answers and the storage charge (cached tokens x full TTL x storage price)."""
    out, storage = [], 0.0
    for s in sessions:
        cache = _rest("POST", "cachedContents", {
            "model": f"models/{MODEL}",
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": s.document}]}],
            "ttl": f"{CACHE_TTL_S}s",
        })
        usage = cache.get("usageMetadata") or cache.get("usage_metadata") or {}
        tokens = int(usage.get("totalTokenCount") or usage.get("cachedInputTokens") or 0)
        storage += tokens / 1e6 * price.cache_storage_per_m_hour * CACHE_TTL_S / 3600
        try:
            for qid, text, _ in s.questions:
                messages = ({"role": "user", "content": f"Question: {text}"},)
                r = send(GeminiTarget.BASE_URL, messages, max_tokens=MAX_TOKENS, timeout_s=60.0,
                         extra_body={**send.keywords["extra_body"],
                                     "extra_body": {"google": {"cached_content": cache["name"]}}})
                meter.add(r, messages, MAX_TOKENS)
                out.append(_answer(qid, r))
        finally:
            _rest("DELETE", cache["name"])
    return out, storage


def run_batch(sessions: list[Session], poll_s: float = 15.0, max_wait_s: float = 3600.0
              ) -> tuple[list[Answer], float, dict[str, int]]:
    """Returns answers, job turnaround in seconds, and token usage from the inline responses."""
    requests = [
        {"request": {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": f"{s.document}\n\nQuestion: {text}"}]}],
            "generation_config": {"max_output_tokens": MAX_TOKENS, "temperature": 0,
                                  "thinking_config": {"thinking_level": "minimal"}},
        }, "metadata": {"key": qid}}
        for s in sessions for qid, text, _ in s.questions
    ]
    started = time.perf_counter()
    job = _rest("POST", f"models/{MODEL}:batchGenerateContent",
                {"batch": {"display_name": "maxionbench-e6", "input_config": {"requests": {"requests": requests}}}})
    name = job["name"]
    while not job.get("done"):
        if time.perf_counter() - started > max_wait_s:
            raise TimeoutError(f"batch {name} not done after {max_wait_s:.0f} s")
        time.sleep(poll_s)
        job = _rest("GET", name)
    turnaround = time.perf_counter() - started
    if "error" in job:
        raise RuntimeError(f"batch {name} failed: {job['error']}")
    responses = _find_inlined(job)
    keys = [r["metadata"]["key"] for r in requests]
    out, usage = [], {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for n, item in enumerate(responses):
        qid = (item.get("metadata") or {}).get("key") or keys[n]
        resp = item.get("response")
        if not resp:
            out.append(Answer(qid, "", "error", None, None, 0, 0, 0))
            continue
        meta = resp.get("usageMetadata") or {}
        text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"].get("parts", []) if not p.get("thought"))
        prompt, cached = int(meta.get("promptTokenCount") or 0), int(meta.get("cachedContentTokenCount") or 0)
        output = int(meta.get("candidatesTokenCount") or 0) + int(meta.get("thoughtsTokenCount") or 0)
        usage["input_tokens"] += prompt
        usage["cached_tokens"] += cached
        usage["output_tokens"] += output
        out.append(Answer(qid, text.strip(), "ok", None, None, prompt, cached, output))
    return out, turnaround, usage


def _find_inlined(node: Any) -> list[dict[str, Any]]:
    """The inline responses list, wherever the operation nests it (response or metadata.output)."""
    if isinstance(node, dict):
        value = node.get("inlinedResponses")
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return _find_inlined(value)
        for child in node.values():
            found = _find_inlined(child)
            if found:
                return found
    return []


def trial_metrics(arm: str, answers: list[Answer], sessions: list[Session], spend: float, turnaround_s: float,
                  storage_usd: float) -> dict[str, float]:
    golds = {qid: gold for s in sessions for qid, _, gold in s.questions}
    ok = [a for a in answers if a.status == "ok"]
    prompt = sum(a.prompt_tokens for a in ok)
    m = {
        "requests": float(len(answers)),
        "errors": float(len(answers) - len(ok)),
        "spend_usd": spend,
        "storage_usd": storage_usd,
        "usd_per_1k_requests": 1000 * spend / len(answers),
        "cached_token_ratio": sum(a.cached_tokens for a in ok) / prompt if prompt else 0.0,
        "prompt_tokens_mean": prompt / len(ok) if ok else 0.0,
        "em": sum(grade_qa(a.text, [golds[a.qid]]).em for a in answers) / len(answers),
    }
    if arm == "batch":
        m["turnaround_s"] = turnaround_s
    for name, values in (("ttft", [a.ttft_s for a in ok]), ("e2e", [a.e2e_s for a in ok])):
        ms = [v * 1000 for v in values if v is not None]
        if ms:
            m.update({f"{name}_{k}": v for k, v in latency_summary(ms).items()})
    return m


def run_e6(arms: list[str], n_sessions: int, questions: int, distractors: int, repeats: int, seed: int,
           out_root: Path, log: Callable[[str], None]) -> Path:
    spec = {"name": "e6-gemini-caching", "model": MODEL, "arms": arms, "sessions": n_sessions, "questions": questions,
            "distractors": distractors, "repeats": repeats, "seed": seed, "max_tokens": MAX_TOKENS,
            "cache_ttl_s": CACHE_TTL_S, "reasoning_effort": "minimal"}
    started_at = utc_now_iso()
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-e6-gemini-caching"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    target = GeminiTarget({"model": MODEL, "reasoning_effort": "minimal"})
    _, price, _ = target.pricing()
    send = functools.partial(chat_completion, **target.request_options())
    scrub, key_present = _scrubber()
    trials, rows = [], []
    for rep in range(repeats):
        for a_i, arm in enumerate(arms):
            sessions = build_sessions(n_sessions, questions, distractors, seed=seed * 1000 + rep * 10 + a_i)
            n_req = n_sessions * questions
            est_in = sum(len(s.document) // 3 + 200 for s in sessions) * questions

            def estimate(p: ModelPrice) -> float:
                return cost_usd(p, input_tokens=est_in, output_tokens=n_req * MAX_TOKENS) + 0.01  # + storage

            t0 = time.perf_counter()
            label = f"e6/{arm}/r{rep}"
            with metered(target, estimate, label) as meter:
                turnaround, storage = 0.0, 0.0
                if arm == "implicit":
                    answers, _ = run_implicit(sessions, send, meter)
                    spend = meter.spend_usd
                elif arm == "explicit":
                    answers, storage = run_explicit(sessions, send, meter, price)
                    spend = meter.spend_usd + storage
                    meter.usage["storage_usd_micros"] = round(storage * 1e6)
                else:
                    answers, turnaround, usage = run_batch(sessions)
                    batch_price = ModelPrice(price.batch_input_per_m, price.batch_output_per_m,
                                             price.cached_input_per_m / 2, 0.0, 0.0, 0.0)
                    spend = cost_usd(batch_price, input_tokens=usage["input_tokens"],
                                     output_tokens=usage["output_tokens"], cached_tokens=usage["cached_tokens"])
                    meter.usage.update(usage)
                meter.override_usd = spend  # commit the arm's full cost (storage, batch price)
            metrics = trial_metrics(arm, answers, sessions, spend, turnaround, storage)
            log(f"{label}: ${spend:.4f} cached={metrics['cached_token_ratio']:.2f} em={metrics['em']:.2f}")
            rows += [{"arm": arm, "repeat": rep, **a.__dict__} for a in answers]
            trials.append(TrialResult(
                trial_id=f"{arm}-r{rep}", cell_id=arm, repeat=rep, seed=seed * 1000 + rep * 10 + a_i, status="ok",
                started_at=started_at, duration_s=round(time.perf_counter() - t0, 3), host_load_1m_before=0.0,
                quiet_host_ok=True, metrics=metrics, requests_per_endpoint=[len(answers)],
                target={**target.describe(), "arm": arm}, error=None))
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(scrub(json.dumps(row, ensure_ascii=False)) + "\n")
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"], description=__doc__.split("\n\n")[0],
        spec=spec,
        provenance=Provenance(
            git_commit=_git(["rev-parse", "HEAD"]) or "unknown", git_dirty=bool(_git(["status", "--porcelain"])),
            spec_fingerprint=stable_config_fingerprint(spec), started_at=started_at, finished_at=utc_now_iso(),
            host=collect_system_info(),
            tools={"python": platform.python_version(), "harness_result_schema": RESULT_SCHEMA_VERSION,
                   "gemini_key_present": key_present, "trials_planned": len(trials), "trials_completed": len(trials)}),
        trials=trials, cells=aggregate_cells(trials, {arm: {"arm": arm} for arm in arms}))
    (out_dir / "results.json").write_text(scrub(json.dumps(result.to_dict(), indent=2)) + "\n", encoding="utf-8")
    log(f"wrote {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="E6 Gemini caching economics")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/e6"))
    args = parser.parse_args(argv)
    arms = args.arms.split(",")
    if set(arms) - set(ARMS):
        raise SystemExit(f"arms must be among {ARMS}")
    run_e6(arms, args.sessions, args.questions, args.distractors, args.repeats, args.seed, args.out,
           lambda m: print(f"[e6] {m}", file=sys.stderr, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
