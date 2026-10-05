"""v0.2 CPU serving benchmark: routing policy x offered load over llama.cpp replicas.

Starts N CPU-only `llama-server` replicas, replays multi-turn RAG sessions (shared document
context, several follow-up questions) as an open-loop Poisson stream, and compares endpoint
pickers on goodput at SLO, TTFT tails, and prefix-cache hit ratio. An optional fault run kills
one replica mid-schedule to measure failover and graceful degradation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterator
import urllib.request

from maxionbench.rag.loadgen import RequestRecord, RequestSpec, run_open_loop, summarize
from maxionbench.rag.routing import PICKERS
from maxionbench.tools.rag_eval import build_messages

FOLLOW_UPS = (
    "Which document numbers support your answer? Reply with the numbers only.",
    "Quote the single most relevant sentence from the documents.",
    "Name one entity from the documents that is relevant to the question.",
)


def build_workload(dataset_dir: Path, *, sessions: int, turns: int, k: int, window: int, seed: int) -> list[RequestSpec]:
    """Sessions share a k-paragraph context (gold evidence + random distractors) across turns."""
    rng = random.Random(seed)
    docs: dict[str, str] = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs[row["doc_id"]] = row["text"]
    gold: dict[str, list[str]] = {}
    with (dataset_dir / "qrels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            gold.setdefault(qid, []).append(doc_id)
    questions = []
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["query_id"] in gold:
                questions.append(row)
    doc_ids = list(docs)
    chosen = rng.sample(questions, sessions)
    per_session: list[list[RequestSpec]] = []
    for s, q in enumerate(chosen):
        ctx_ids = gold[q["query_id"]][:k]
        ctx_ids += [d for d in rng.sample(doc_ids, k) if d not in ctx_ids][: k - len(ctx_ids)]
        rng.shuffle(ctx_ids)
        ctx = [docs[d] for d in ctx_ids]
        prompts = [q["text"], *FOLLOW_UPS][:turns]
        per_session.append(
            [
                RequestSpec(
                    request_id=f"s{s}-t{t}",
                    session_id=f"s{s}",
                    prefix_key=f"s{s}",
                    messages=tuple(build_messages(prompt, ctx)),
                    fallback_text=ctx[0][:200],  # retrieval-only degraded answer
                )
                for t, prompt in enumerate(prompts)
            ]
        )
    # Interleave: `window` sessions are active at once; each request is the next turn of a random active session.
    order: list[RequestSpec] = []
    pending = list(range(sessions))
    active = [pending.pop(0) for _ in range(min(window, sessions))]
    cursor = [0] * sessions
    while active:
        s = rng.choice(active)
        order.append(per_session[s][cursor[s]])
        cursor[s] += 1
        if cursor[s] == len(per_session[s]):
            active.remove(s)
            if pending:
                active.append(pending.pop(0))
    return order


@contextmanager
def replicas(args: Any, log_dir: Path) -> Iterator[list[subprocess.Popen[bytes]]]:
    procs = []
    try:
        for i in range(args.replicas):
            log = (log_dir / f"replica{i}.log").open("wb")
            cmd = [
                args.llama_server, "-m", str(Path(args.model).expanduser()),
                "--device", "none", "-ngl", "0", "--no-op-offload",  # CPU only, no Metal offload
                "-t", str(args.threads), "-np", str(args.slots), "-c", str(args.ctx),
                "--cache-ram", str(args.cache_ram_mib), "--metrics",
                "--host", "127.0.0.1", "--port", str(args.base_port + i),
            ]
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        for i in range(args.replicas):
            _wait_healthy(f"http://127.0.0.1:{args.base_port + i}", timeout_s=120)
        yield procs
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _wait_healthy(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"replica at {url} did not become healthy")


def _split_summary(records: list[RequestRecord], at_s: float, args: Any) -> dict[str, Any]:
    before = [r for r in records if r.scheduled_s < at_s]
    after = [r for r in records if r.scheduled_s >= at_s]
    end = max(r.scheduled_s for r in records)
    return {
        "before_fault": summarize(before, duration_s=at_s, ttft_slo_s=args.ttft_slo_s, e2e_slo_s=args.e2e_slo_s),
        "after_fault": summarize(after, duration_s=end - at_s, ttft_slo_s=args.ttft_slo_s, e2e_slo_s=args.e2e_slo_s),
    }


def run(args: Any) -> dict[str, Any]:
    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    specs = build_workload(
        Path(args.dataset), sessions=args.sessions, turns=args.turns, k=args.k, window=args.window, seed=args.seed
    )
    urls = [f"http://127.0.0.1:{args.base_port + i}" for i in range(args.replicas)]
    plan: list[tuple[str, float, bool]] = [(p, r, False) for r in map(float, args.rates.split(",")) for p in args.policies.split(",")]
    if args.fault_rate > 0:
        plan += [(p, args.fault_rate, True) for p in args.fault_policies.split(",")]

    results = []
    records_by_run: dict[tuple[str, float, bool], list[RequestRecord]] = {}
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for policy, rate, fault in plan:
            with replicas(args, out_dir / "logs") as procs:  # fresh replicas: no cache carry-over between runs
                picker = PICKERS[policy](args.replicas)
                fault_at = 0.4 * len(specs) / rate
                events = [(fault_at, procs[0].kill)] if fault else []
                records, duration = run_open_loop(
                    specs, base_urls=urls, picker=picker, rate_rps=rate, max_in_flight=args.max_in_flight,
                    timeout_s=args.timeout_s, max_tokens=args.max_tokens, seed=args.seed, events=events,
                )
            row: dict[str, Any] = {"policy": policy, "rate_rps": rate, "fault": fault}
            row.update(summarize(records, duration_s=duration, ttft_slo_s=args.ttft_slo_s, e2e_slo_s=args.e2e_slo_s))
            row["requests_per_replica"] = [sum(1 for r in records if r.endpoint == i) for i in range(args.replicas)]
            records_by_run[(policy, rate, fault)] = records
            if fault:
                row["fault_at_s"] = round(fault_at, 2)
                row.update(_split_summary(records, fault_at, args))
                baseline = records_by_run.get((policy, rate, False))
                if baseline:  # same seed -> same arrival schedule; compare identical time windows
                    row["no_fault_baseline"] = _split_summary(baseline, fault_at, args)
            results.append(row)
            for r in records:
                fh.write(json.dumps({"policy": policy, "rate_rps": rate, "fault": fault, **asdict(r)}) + "\n")
            print(f"[serving-bench] {json.dumps(row)}", file=sys.stderr, flush=True)

    summary = {
        "profile": "maxionbench-v0.2-cpu-serving",
        "config": {k: v for k, v in vars(args).items()},
        "llama_server_version": subprocess.run(
            [args.llama_server, "--version"], capture_output=True, text=True, check=False
        ).stderr.strip().splitlines()[-2:],
        "requests_per_run": len(specs),
        "runs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/processed/hotpot_portable")
    parser.add_argument("--model", default="~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--cache-ram-mib", type=int, default=64)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--policies", default="round_robin,least_outstanding,prefix_affinity")
    parser.add_argument("--rates", default="0.5,1,2")
    parser.add_argument("--fault-rate", type=float, default=0.5)
    parser.add_argument("--fault-policies", default="least_outstanding,prefix_affinity")
    parser.add_argument("--max-in-flight", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--ttft-slo-s", type=float, default=2.5)
    parser.add_argument("--e2e-slo-s", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/v0.2/serving")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
