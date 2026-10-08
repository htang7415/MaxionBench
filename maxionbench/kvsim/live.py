"""Replay AgentX agent sessions live through a serving target (llm-d over inference-sim workers).

The offline simulator (sim.py) prices cache policies without latency. This replays the same sessions
in wall-clock time through the real llm-d router, so routing decisions come from the EPP and latency
from the workers. Each 64-token trace block becomes `tokens_per_block` token ids derived from
(session, block id), so shared prefixes in the trace are shared token prefixes on the wire; requests
are /v1/completions with token-id prompts. `time_scale` > 1 compresses the trace's clock (arrivals and
gaps); worker latencies in the spec must be divided by the same factor to keep the system consistent.

    python -m maxionbench.kvsim.live experiments/k5_llmd_tier_aware.yaml [--out results]

Open loop: arrivals follow the trace regardless of how fast responses come back, and latency is
measured from the scheduled arrival (no coordinated omission).
"""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import heapq
import http.client
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np
import yaml

from maxionbench.datasets.sources import verified_path
from maxionbench.harness.llmd import LlmdNoK8s
from maxionbench.harness.provenance import make_provenance
from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells
from maxionbench.kvsim.traces import TRACE_FILE, Request, Session, load_sessions
from maxionbench.schemas.result_schema import utc_now_iso

SPEC_SCHEMA = "maxionbench-kvlive-v1"
VOCAB = 150_000  # token ids stay inside a Qwen-sized vocabulary
_MIX = np.uint64(0x9E3779B97F4A7C15)


@dataclass(frozen=True)
class Arrival:
    t: float  # seconds from replay start (already time-scaled)
    session: int
    request: Request


def schedule(sessions: Sequence[Session], concurrency: int, horizon_s: float, stagger_s: float, time_scale: float,
             seed: int) -> list[Arrival]:
    """Arrivals before `horizon_s` with `concurrency` sessions active: the first ones start uniformly in
    [0, stagger_s), and a session that ends is replaced by the next (sessions shuffled by `seed`).
    Trace times are divided by `time_scale`."""
    rng = random.Random(seed)
    order = list(range(len(sessions)))
    rng.shuffle(order)
    pending = iter(order)
    ends: list[tuple[float, int]] = []
    out: list[Arrival] = []

    def start(s_idx: int, t0: float) -> None:
        s = sessions[s_idx]
        out.extend(Arrival(t0 + r.t / time_scale, s_idx, r) for r in s.requests if t0 + r.t / time_scale < horizon_s)
        heapq.heappush(ends, (t0 + s.span / time_scale, s_idx))

    for s_idx in (next(pending, None) for _ in range(concurrency)):
        if s_idx is None:
            break
        start(s_idx, rng.uniform(0.0, stagger_s))
    while ends:
        t_end, _ = heapq.heappop(ends)
        nxt = next(pending, None)
        if t_end >= horizon_s or nxt is None:
            continue
        start(nxt, t_end)
    out.sort(key=lambda a: a.t)
    return out


def prompt_tokens(session: int, blocks: np.ndarray, per_block: int) -> list[int]:
    """Deterministic token ids for a request's blocks: equal (session, block) -> equal tokens."""
    keys = (np.uint64(session) << np.uint64(32)) | blocks.astype(np.uint64)
    x = keys[:, None] * np.uint64(per_block) + np.arange(per_block, dtype=np.uint64)[None, :]
    x = (x + _MIX) * _MIX  # wraps mod 2**64
    x ^= x >> np.uint64(31)
    return (x % np.uint64(VOCAB - 1000) + np.uint64(1000)).ravel().tolist()


@dataclass
class Outcome:
    scheduled_s: float
    status: str
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    error: str | None = None


def stream_completion(base_url: str, body: dict[str, Any], timeout_s: float) -> tuple[float | None, dict[str, Any]]:
    """POST a streaming /v1/completions; returns (seconds to the first text chunk, usage). Raises on failure."""
    parts = urlsplit(base_url)
    conn = http.client.HTTPConnection(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    started = time.perf_counter()
    try:
        conn.request("POST", "/v1/completions", body=json.dumps(body), headers={"content-type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"http {resp.status}: {resp.read(300).decode('utf-8', 'replace')}")
        ttft, usage = None, {}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if ttft is None and any(c.get("text") for c in chunk.get("choices") or []):
                ttft = time.perf_counter() - started
            usage = chunk.get("usage") or usage
        return ttft, usage
    finally:
        conn.close()


def replay(arrivals: Sequence[Arrival], base_url: str, model: str, *, tokens_per_block: int, output_scale: float,
           max_output_tokens: int, timeout_s: float) -> list[Outcome]:
    outcomes: list[Outcome | None] = [None] * len(arrivals)
    t0 = time.perf_counter()

    def send(i: int, a: Arrival) -> None:
        sched_abs = t0 + a.t
        try:
            body = {"model": model, "prompt": prompt_tokens(a.session, a.request.blocks, tokens_per_block),
                    "max_tokens": max(1, min(max_output_tokens, round(a.request.out_tokens * output_scale))),
                    "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
            sent = time.perf_counter()
            ttft, usage = stream_completion(base_url, body, timeout_s)
            done = time.perf_counter()
            queued = sent - sched_abs  # client-side lateness counts against latency
            outcomes[i] = Outcome(
                scheduled_s=a.t, status="ok", ttft_s=None if ttft is None else queued + ttft, e2e_s=done - sched_abs,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)))
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none stops the replay
            outcomes[i] = Outcome(scheduled_s=a.t, status="error", error=str(exc)[:200])

    with ThreadPoolExecutor(max_workers=512) as pool:
        for i, a in enumerate(arrivals):
            delay = t0 + a.t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(send, i, a)
    return [o for o in outcomes if o is not None]


def trial_metrics(outcomes: Sequence[Outcome], warmup_s: float, ttft_slo_s: float) -> dict[str, float]:
    window = [o for o in outcomes if o.scheduled_s >= warmup_s]
    ok = [o for o in window if o.status == "ok"]
    if not ok:
        raise RuntimeError(f"no successful requests after warm-up ({len(window)} attempted)")
    ttfts = [o.ttft_s for o in ok if o.ttft_s is not None]
    prompt = sum(o.prompt_tokens for o in ok)
    cached = sum(o.cached_tokens for o in ok)
    span = max(o.scheduled_s for o in window) - warmup_s
    p50, p90, p99 = (float(v) for v in np.percentile(ttfts, [50, 90, 99]))
    return {
        "requests_measured": float(len(window)),
        "error_rate": 1 - len(ok) / len(window),
        "ttft_p50_s": p50, "ttft_p90_s": p90, "ttft_p99_s": p99,
        "ttft_mean_s": sum(ttfts) / len(ttfts),
        "e2e_p50_s": float(np.percentile([o.e2e_s for o in ok if o.e2e_s is not None], 50)),
        "slo_attainment": sum(1 for t in ttfts if t <= ttft_slo_s) / len(window),
        "goodput_rps": sum(1 for t in ttfts if t <= ttft_slo_s) / span if span > 0 else float("nan"),
        "cached_token_share": cached / prompt if prompt else 0.0,
        "recomputed_tokens_per_request": (prompt - cached) / len(ok),
    }


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    for key in ("name", "replay", "target", "cells"):
        if key not in spec:
            raise ValueError(f"{path}: missing {key}")
    return spec


def target_params(target: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """llm-d target params for a cell: scorer profile plus the workers' GPU/CPU KV sizes (inference-sim
    workers), or native vllm-metal workers configured by `worker_params`."""
    t = {**target, **cell}
    if t.get("workers") == "vllm_metal":
        return {"workers": "vllm_metal", "scorer_profile": t["scorer_profile"], "model": t.get("model", "qwen3"),
                "worker_params": dict(t["worker_params"])}
    args = [*t.get("sim_args", []), "--enable-kvcache", "--kv-cache-size", str(t["gpu_kv_blocks"]),
            "--cpu-kv-cache-size", str(t.get("cpu_kv_blocks", 0))]
    return {"workers": "sim", "scorer_profile": t["scorer_profile"], "model": t.get("model", "qwen3"),
            "worker_params": {"replicas": t["replicas"], "base_port": t.get("base_port", 8300),
                              "image": t["image"], "args": args}}


def run(spec_path: Path, out_root: Path, log=lambda m: print(m, file=sys.stderr)) -> Path:
    spec = load_spec(spec_path)
    rp = spec["replay"]
    started_at = utc_now_iso()
    traces = spec.get("traces") or {"default": spec.get("trace", TRACE_FILE)}  # e.g. one trace per context policy
    sessions = {name: load_sessions(verified_path(rel), idle_cap_s=float(spec.get("idle_cap_s", 300)))
                for name, rel in traces.items()}
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    cells = {"/".join(f"{k}={v}" for k, v in c.items()): c for c in spec["cells"]}
    if "traces" in spec:
        cells = {f"trace={name}/{cell_id}": {"trace": name, **c} for name in traces for cell_id, c in cells.items()}
    trials: list[TrialResult] = []
    for rep in range(int(spec.get("repeats", 1))):
        seed = int(spec.get("seed", 0)) * 1000 + rep
        arrivals = {name: schedule(s, int(rp["concurrency"]), float(rp["horizon_s"]), float(rp["warmup_s"]),
                                   float(rp.get("time_scale", 1.0)), seed) for name, s in sessions.items()}
        for cell_id, cell in cells.items():  # every cell replays the same session schedule within a repeat
            params = target_params(spec["target"], {k: v for k, v in cell.items() if k != "trace"})
            trace_arrivals = arrivals[cell.get("trace", "default")]
            label = f"{cell_id}/r{rep}"
            log(f"{label}: {len(trace_arrivals)} arrivals over {rp['horizon_s']} s")
            t_start, trial_started = time.perf_counter(), utc_now_iso()
            target = LlmdNoK8s(params, out_dir / "logs" / label.replace("/", "_"))
            with target:
                outcomes = replay(trace_arrivals, target.base_urls[0], params["model"],
                                  tokens_per_block=int(rp["tokens_per_block"]), output_scale=float(rp["output_scale"]),
                                  max_output_tokens=int(rp["max_output_tokens"]), timeout_s=float(rp["timeout_s"]))
                server = target.collect()
            metrics = trial_metrics(outcomes, float(rp["warmup_s"]), float(rp["ttft_slo_s"]))
            per_worker = server["requests_per_worker"]
            metrics.update({
                "server_gpu_hit_ratio": server["server_prefix_cache_hit_ratio"] or 0.0,
                "server_cpu_loaded_ratio": server["server_cpu_loaded_ratio"] or 0.0,
                "load_imbalance": max(per_worker) / (sum(per_worker) / len(per_worker)) if sum(per_worker) else 0.0,
            })
            log(f"{label}: " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            trials.append(TrialResult(
                trial_id=label, cell_id=cell_id, repeat=rep, seed=seed, status="ok", started_at=trial_started,
                duration_s=round(time.perf_counter() - t_start, 3), host_load_1m_before=0.0, quiet_host_ok=True,
                metrics={k: round(v, 6) for k, v in metrics.items()}, requests_per_endpoint=[int(x) for x in per_worker],
                target=target.describe(), error=None))
            with (out_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
                for o in outcomes:
                    fh.write(json.dumps({"trial": label, **o.__dict__}) + "\n")
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"], description=spec.get("description", ""),
        spec=spec, provenance=make_provenance(spec, started_at, {
            "traces": traces, "trials_completed": len(trials)}),
        trials=trials, cells=aggregate_cells(trials, cells))
    (out_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="python -m maxionbench.kvsim.live", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    run(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
