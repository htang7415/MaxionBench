"""RPC baseline measurement helpers."""

from __future__ import annotations

import time
from typing import Callable

from maxionbench.metrics.latency import percentile_ms


def measure_rpc_baseline(request_fn: Callable[[], None], request_count: int = 1000) -> dict[str, float]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    samples_ms: list[float] = []
    for _ in range(request_count):
        start = time.perf_counter()
        request_fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "rtt_baseline_ms_p50": percentile_ms(samples_ms, 50),
        "rtt_baseline_ms_p99": percentile_ms(samples_ms, 99),
    }
