from __future__ import annotations

import numpy as np
import pytest

from maxionbench.adapters.mock import MockAdapter
from maxionbench.datasets.d3_generator import D3Params
from maxionbench.scenarios.s3_churn_smooth import S3Config, run as run_s3
from maxionbench.scenarios.s3b_churn_bursty import S3bConfig, _mean_write_multiplier_normalizer


class _CountingAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.optimize_calls = 0

    def optimize_or_compact(self) -> None:
        self.optimize_calls += 1


def test_s3_burst_clock_resets_at_measurement_start() -> None:
    adapter = MockAdapter()
    adapter.create(collection="s3-clock", dimension=8, metric="ip")
    cfg = S3Config(
        vector_dim=8,
        num_vectors=200,
        num_queries=10,
        top_k=10,
        sla_threshold_ms=120.0,
        warmup_s=0.5,
        steady_state_s=0.5,
        lambda_req_s=20.0,
        read_rate=16.0,
        insert_rate=2.0,
        update_rate=1.0,
        delete_rate=1.0,
        maintenance_interval_s=60.0,
        phase_timing_mode="bounded",
        max_events=200,
    )
    d3_params = D3Params(
        k_clusters=32,
        num_tenants=12,
        num_acl_buckets=8,
        num_time_buckets=12,
        beta_tenant=0.75,
        beta_acl=0.70,
        beta_time=0.65,
        seed=9,
    )

    seen_clock: list[float] = []

    def _capture_clock(sim_t: float) -> float:
        seen_clock.append(float(sim_t))
        return 1.0

    run_s3(
        adapter=adapter,
        cfg=cfg,
        rng=np.random.default_rng(9),
        d3_params=d3_params,
        burst_multiplier_fn=_capture_clock,
    )

    assert seen_clock
    resets = [idx for idx in range(1, len(seen_clock)) if seen_clock[idx] < seen_clock[idx - 1]]
    assert len(resets) == 1
    assert seen_clock[0] == pytest.approx(0.0, abs=1e-12)
    assert seen_clock[resets[0]] == pytest.approx(0.0, abs=1e-12)


def test_s3_phase_clock_advances_for_read_events_and_triggers_maintenance() -> None:
    adapter = _CountingAdapter()
    adapter.create(collection="s3-clock-maintenance", dimension=8, metric="ip")
    cfg = S3Config(
        vector_dim=8,
        num_vectors=200,
        num_queries=10,
        top_k=10,
        sla_threshold_ms=120.0,
        warmup_s=0.0,
        steady_state_s=1.0,
        lambda_req_s=20.0,
        read_rate=16.0,
        insert_rate=2.0,
        update_rate=1.0,
        delete_rate=1.0,
        maintenance_interval_s=0.1,
        phase_timing_mode="strict",
        max_events=200,
    )
    d3_params = D3Params(
        k_clusters=32,
        num_tenants=12,
        num_acl_buckets=8,
        num_time_buckets=12,
        beta_tenant=0.75,
        beta_acl=0.70,
        beta_time=0.65,
        seed=11,
    )
    run_s3(
        adapter=adapter,
        cfg=cfg,
        rng=np.random.default_rng(11),
        d3_params=d3_params,
    )
    # strict mode => measure events = int(lambda * steady_state) = 20, dt = 0.05
    # maintenance fires at sim_t in {0.1, 0.2, ..., 0.9} => 9 times
    assert adapter.optimize_calls == 9


def test_s3b_write_multiplier_normalizer_preserves_mean_baseline() -> None:
    cfg = S3bConfig(
        base=S3Config(
            vector_dim=8,
            num_vectors=200,
            num_queries=10,
            top_k=10,
            sla_threshold_ms=120.0,
            warmup_s=0.0,
            steady_state_s=1.0,
            lambda_req_s=1000.0,
            read_rate=800.0,
            insert_rate=100.0,
            update_rate=50.0,
            delete_rate=50.0,
            maintenance_interval_s=60.0,
            phase_timing_mode="strict",
            max_events=1000,
        ),
        on_s=30.0,
        off_s=90.0,
        on_write_mult=8.0,
        off_write_mult=0.25,
    )
    normalizer = _mean_write_multiplier_normalizer(cfg)
    assert normalizer == pytest.approx(2.1875)
    weighted_mean = ((cfg.on_s * (cfg.on_write_mult / normalizer)) + (cfg.off_s * (cfg.off_write_mult / normalizer))) / (
        cfg.on_s + cfg.off_s
    )
    assert weighted_mean == pytest.approx(1.0)
