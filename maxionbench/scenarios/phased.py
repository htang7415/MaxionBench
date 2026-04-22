"""Warmup + measurement phase execution helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class PhaseStats:
    target_s: float
    elapsed_s: float
    requests: int


def run_query_phases(
    *,
    total_queries: int,
    clients_read: int,
    warmup_s: float,
    steady_state_s: float,
    evaluate_query: Callable[[int], T],
    strict_timing: bool = False,
    max_requests_per_phase: int | None = None,
) -> tuple[list[tuple[int, T]], PhaseStats, PhaseStats]:
    if total_queries < 1:
        raise ValueError("total_queries must be >= 1")
    workers = max(1, clients_read)
    cursor = 0

    def next_index() -> int:
        nonlocal cursor
        idx = cursor % total_queries
        cursor += 1
        return idx

    cap = None
    if max_requests_per_phase is not None:
        cap = int(max_requests_per_phase)
        if cap < 1:
            cap = None

    def run_phase(target_s: float, *, collect: bool) -> tuple[list[tuple[int, T]], PhaseStats]:
        collected: list[tuple[int, T]] = []
        target = max(0.0, float(target_s))
        requests = 0
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) if workers > 1 else _null_pool() as pool:
            while True:
                elapsed = time.perf_counter() - started
                if requests > 0:
                    if target > 0.0 and elapsed >= target:
                        break
                    if not strict_timing and requests >= total_queries:
                        break
                    if cap is not None and requests >= cap:
                        break

                batch_size = workers
                if not strict_timing:
                    batch_size = min(batch_size, total_queries - requests)
                if cap is not None:
                    batch_size = min(batch_size, cap - requests)
                if batch_size < 1:
                    break
                query_indices = [next_index() for _ in range(batch_size)]
                if workers == 1:
                    outputs = [evaluate_query(query_indices[0])]
                else:
                    outputs = list(pool.map(evaluate_query, query_indices))
                requests += batch_size
                if collect:
                    collected.extend(list(zip(query_indices, outputs)))
        elapsed = time.perf_counter() - started
        return (
            collected,
            PhaseStats(target_s=target, elapsed_s=float(elapsed), requests=int(requests)),
        )

    _, warmup_stats = run_phase(warmup_s, collect=False)
    measured, measure_stats = run_phase(steady_state_s, collect=True)
    return measured, warmup_stats, measure_stats


class _null_pool(Generic[T]):
    def __enter__(self) -> "_null_pool[T]":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        return None

    def map(self, fn: Callable[[int], T], values: list[int]) -> list[T]:
        return [fn(v) for v in values]
