"""Latency metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def percentile_ms(samples_ms: Sequence[float], percentile: float) -> float:
    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")
    arr = np.asarray(samples_ms, dtype=np.float64)
    return float(np.percentile(arr, percentile))


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile_ms(samples_ms, 50),
        "p95_ms": percentile_ms(samples_ms, 95),
        "p99_ms": percentile_ms(samples_ms, 99),
    }
