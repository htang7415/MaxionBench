from __future__ import annotations

import pytest

from maxionbench.metrics.cost_rhu import RHUReferences, RHUWeights
from maxionbench.metrics.latency import latency_summary, percentile_ms
from maxionbench.metrics.resources import profile_from_adapter_stats, rhu_rate_for_profile
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import p99_inflation, sla_violation_rate
from maxionbench.schemas.adapter_contract import AdapterStats


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


def test_percentile_ms_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples_ms"):
        percentile_ms([], 99)


def test_robustness_metrics() -> None:
    assert p99_inflation(20.0, 10.0) == 2.0
    assert p99_inflation(20.0, 0.0) == 0.0
    assert sla_violation_rate(total_requests=100, over_sla=3, errors=2) == 0.05


def test_robustness_metrics_reject_negative_baseline() -> None:
    with pytest.raises(ValueError, match="baseline_p99_ms"):
        p99_inflation(20.0, -1.0)


def test_resource_profile_and_rhu_rate() -> None:
    stats = AdapterStats(
        vector_count=1000,
        deleted_count=10,
        index_size_bytes=1024,
        ram_usage_bytes=8 * 1024**3,
        disk_usage_bytes=2 * 1024**4,
        engine_uptime_s=12.0,
    )
    profile = profile_from_adapter_stats(stats=stats, client_count=16, gpu_count=1.0)
    assert profile.cpu_vcpu == 16.0
    assert profile.gpu_count == 1.0
    assert profile.ram_gib == 8.0
    assert profile.disk_tb == 2.0

    rate = rhu_rate_for_profile(profile=profile, refs=RHUReferences(), weights=RHUWeights())
    expected = 0.25 * (16.0 / 96.0) + 0.25 * (1.0 / 1.0) + 0.25 * (8.0 / 512.0) + 0.25 * (2.0 / 7.68)
    assert rate == pytest.approx(expected)


def test_resource_profile_rejects_negative_client_count() -> None:
    stats = AdapterStats(
        vector_count=0,
        deleted_count=0,
        index_size_bytes=0,
        ram_usage_bytes=0,
        disk_usage_bytes=0,
        engine_uptime_s=0.0,
    )
    with pytest.raises(ValueError, match="client_count"):
        profile_from_adapter_stats(stats=stats, client_count=-1)


def test_resource_profile_rejects_negative_resource_values() -> None:
    bad_ram = AdapterStats(
        vector_count=0,
        deleted_count=0,
        index_size_bytes=0,
        ram_usage_bytes=-1,
        disk_usage_bytes=0,
        engine_uptime_s=0.0,
    )
    with pytest.raises(ValueError, match="ram_usage_bytes"):
        profile_from_adapter_stats(stats=bad_ram, client_count=1)

    bad_disk = AdapterStats(
        vector_count=0,
        deleted_count=0,
        index_size_bytes=0,
        ram_usage_bytes=0,
        disk_usage_bytes=-1,
        engine_uptime_s=0.0,
    )
    with pytest.raises(ValueError, match="disk_usage_bytes"):
        profile_from_adapter_stats(stats=bad_disk, client_count=1)

    ok_stats = AdapterStats(
        vector_count=0,
        deleted_count=0,
        index_size_bytes=0,
        ram_usage_bytes=0,
        disk_usage_bytes=0,
        engine_uptime_s=0.0,
    )
    with pytest.raises(ValueError, match="gpu_count"):
        profile_from_adapter_stats(stats=ok_stats, client_count=1, gpu_count=-1.0)
