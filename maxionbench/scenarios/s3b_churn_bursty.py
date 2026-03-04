"""S3b dynamic churn bursty ON/OFF workload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from maxionbench.datasets.d3_generator import D3Params
from maxionbench.scenarios.s3_churn_smooth import S3Config, S3Result, run as run_s3


@dataclass(frozen=True)
class S3bConfig:
    base: S3Config
    on_s: float = 30.0
    off_s: float = 90.0
    on_write_mult: float = 8.0
    off_write_mult: float = 0.25


def run(
    adapter: Any,
    cfg: S3bConfig,
    rng: np.random.Generator,
    *,
    d3_params: D3Params,
) -> S3Result:
    cycle = cfg.on_s + cfg.off_s

    def burst_multiplier(sim_t: float) -> float:
        phase = sim_t % cycle
        if phase < cfg.on_s:
            return cfg.on_write_mult
        return cfg.off_write_mult

    result = run_s3(
        adapter=adapter,
        cfg=cfg.base,
        rng=rng,
        d3_params=d3_params,
        burst_multiplier_fn=burst_multiplier,
    )
    return S3Result(
        p50_ms=result.p50_ms,
        p95_ms=result.p95_ms,
        p99_ms=result.p99_ms,
        qps=result.qps,
        recall_at_10=result.recall_at_10,
        ndcg_at_10=result.ndcg_at_10,
        mrr_at_10=result.mrr_at_10,
        sla_violation_rate=result.sla_violation_rate,
        errors=result.errors,
        info_json=result.info_json,
        measured_requests=result.measured_requests,
        measured_elapsed_s=result.measured_elapsed_s,
        warmup_requests=result.warmup_requests,
        warmup_elapsed_s=result.warmup_elapsed_s,
    )
