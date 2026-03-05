"""RPC baseline measurement helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

from maxionbench.metrics.latency import percentile_ms
from maxionbench.schemas.adapter_contract import QueryRequest


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


def minimal_rpc_request_fn(*, adapter: Any, vector_dim: int, top_k: int = 1) -> Callable[[], None]:
    """Return pinned minimal request: healthcheck + one minimal vector query."""

    if vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    request = QueryRequest(vector=[0.0] * int(vector_dim), top_k=max(1, int(top_k)))

    def _request() -> None:
        adapter.healthcheck()
        adapter.query(request)

    return _request
