from __future__ import annotations

from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import p99_inflation, sla_violation_rate


def test_quality_metrics() -> None:
    retrieved = ["a", "b", "c", "d"]
    gt = ["a", "d", "e"]
    assert recall_at_k(retrieved, gt, k=3) == 1 / 3
    assert mrr_at_k(retrieved, gt, k=10) == 1.0
    assert ndcg_at_10(retrieved, {"a": 3.0, "d": 1.0}) > 0.0


def test_latency_summary() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["p50_ms"] == 3.0
    assert summary["p99_ms"] >= summary["p95_ms"] >= summary["p50_ms"]


def test_robustness_metrics() -> None:
    assert p99_inflation(20.0, 10.0) == 2.0
    assert sla_violation_rate(total_requests=100, over_sla=3, errors=2) == 0.05
