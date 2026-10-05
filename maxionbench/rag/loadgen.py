"""Open-loop load generator for multi-replica LLM serving.

Requests arrive on a seeded Poisson schedule regardless of how fast earlier requests finish,
and latency is measured from the *scheduled* arrival time, so slow responses cannot hide
queueing delay (no coordinated omission). Admission control rejects arrivals when the
in-flight cap is reached; failed or rejected requests fall back to a retrieval-only answer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from maxionbench.metrics.latency import latency_summary
from maxionbench.rag.llm_client import CompletionResult, chat_completion
from maxionbench.rag.routing import EndpointPicker


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    session_id: str
    prefix_key: str
    messages: tuple[Mapping[str, str], ...]
    fallback_text: str = ""


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    session_id: str
    endpoint: int | None
    scheduled_s: float
    status: str  # "ok" | "error" | "timeout" | "rejected" | "no_endpoint"
    ttft_s: float | None  # from scheduled arrival
    e2e_s: float | None  # from scheduled arrival
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    degraded: bool
    error: str | None


def poisson_arrivals(n: int, rate_rps: float, seed: int) -> list[float]:
    if rate_rps <= 0:
        raise ValueError("rate_rps must be > 0")
    rng = random.Random(seed)
    t = 0.0
    arrivals = []
    for _ in range(n):
        t += rng.expovariate(rate_rps)
        arrivals.append(t)
    return arrivals


def run_open_loop(
    specs: Sequence[RequestSpec],
    *,
    base_urls: Sequence[str],
    picker: EndpointPicker,
    rate_rps: float,
    max_in_flight: int,
    timeout_s: float,
    max_tokens: int,
    seed: int,
    send: Callable[..., CompletionResult] = chat_completion,
    events: Sequence[tuple[float, Callable[[], None]]] = (),
) -> tuple[list[RequestRecord], float]:
    """Run the schedule; returns records in arrival order and wall-clock duration in seconds."""
    arrivals = poisson_arrivals(len(specs), rate_rps, seed)
    records: list[RequestRecord | None] = [None] * len(specs)
    lock = threading.Lock()
    in_flight = 0
    t0 = time.perf_counter()

    def worker(i: int, spec: RequestSpec, scheduled_abs: float) -> None:
        nonlocal in_flight
        try:
            idx = picker.acquire(spec.prefix_key)
            if idx is None:
                records[i] = _degraded(spec, None, arrivals[i], "no_endpoint", "no healthy endpoint")
                return
            try:
                result = send(base_urls[idx], spec.messages, max_tokens=max_tokens, timeout_s=timeout_s)
            finally:
                picker.release(idx)
            if result.status == "error" and not (result.error or "").startswith("http "):
                picker.mark_down(idx)  # passive health check: transport failure
            if result.status != "ok":
                records[i] = _degraded(spec, idx, arrivals[i], result.status, result.error)
                return
            offset = time.perf_counter() - result.e2e_s - scheduled_abs  # queueing before send
            records[i] = RequestRecord(
                request_id=spec.request_id,
                session_id=spec.session_id,
                endpoint=idx,
                scheduled_s=arrivals[i],
                status="ok",
                ttft_s=None if result.ttft_s is None else offset + result.ttft_s,
                e2e_s=offset + result.e2e_s,
                prompt_tokens=result.prompt_tokens,
                cached_tokens=result.cached_tokens,
                completion_tokens=result.completion_tokens,
                degraded=False,
                error=None,
            )
        except Exception as exc:  # keep the record; a lost request would bias goodput upward
            records[i] = _degraded(spec, None, arrivals[i], "error", f"{type(exc).__name__}: {exc}")
        finally:
            with lock:
                in_flight -= 1

    timers = [threading.Timer(at_s, fn) for at_s, fn in events]
    for timer in timers:
        timer.start()
    try:
        with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
            for i, spec in enumerate(specs):
                scheduled_abs = t0 + arrivals[i]
                delay = scheduled_abs - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                with lock:
                    admitted = in_flight < max_in_flight
                    if admitted:
                        in_flight += 1
                if not admitted:
                    records[i] = _degraded(spec, None, arrivals[i], "rejected", "admission control")
                    continue
                pool.submit(worker, i, spec, scheduled_abs)
    finally:
        for timer in timers:
            timer.cancel()
    return [r for r in records if r is not None], time.perf_counter() - t0


def summarize(
    records: Sequence[RequestRecord],
    *,
    duration_s: float,
    ttft_slo_s: float,
    e2e_slo_s: float,
) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    good = [
        r
        for r in ok
        if r.ttft_s is not None and r.ttft_s <= ttft_slo_s and r.e2e_s is not None and r.e2e_s <= e2e_slo_s
    ]
    statuses: dict[str, int] = {}
    for r in records:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    prompt = sum(r.prompt_tokens for r in ok)
    out: dict[str, Any] = {
        "requests": len(records),
        "duration_s": round(duration_s, 3),
        "status_counts": statuses,
        "degraded": sum(1 for r in records if r.degraded),
        "slo_attainment": round(len(good) / len(records), 4) if records else 0.0,
        "goodput_rps": round(len(good) / duration_s, 4) if duration_s > 0 else 0.0,
        "prefix_cache_hit_ratio": round(sum(r.cached_tokens for r in ok) / prompt, 4) if prompt else 0.0,
    }
    ttfts = [r.ttft_s * 1000 for r in ok if r.ttft_s is not None]
    if ttfts:
        out["ttft"] = {k: round(v, 1) for k, v in latency_summary(ttfts).items()}
        out["e2e"] = {k: round(v, 1) for k, v in latency_summary([r.e2e_s * 1000 for r in ok]).items()}
    return out


def _degraded(spec: RequestSpec, endpoint: int | None, scheduled_s: float, status: str, error: str | None) -> RequestRecord:
    return RequestRecord(
        request_id=spec.request_id,
        session_id=spec.session_id,
        endpoint=endpoint,
        scheduled_s=scheduled_s,
        status=status,
        ttft_s=None,
        e2e_s=None,
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        degraded=True,
        error=error,
    )
