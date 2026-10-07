from __future__ import annotations

import pytest

from maxionbench.metrics.latency import latency_summary, percentile_ms


def test_latency_summary() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["p50_ms"] == 3.0
    assert summary["p99_ms"] >= summary["p95_ms"] >= summary["p50_ms"]


def test_percentile_ms_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples_ms"):
        percentile_ms([], 99)
