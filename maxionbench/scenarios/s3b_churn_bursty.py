"""S3b dynamic churn bursty ON/OFF workload."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    vectors: np.ndarray | None = None,
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
        vectors=vectors,
    )
    info_payload = _parse_info_payload(result.info_json)
    info_payload["mode"] = "s3_bursty"
    info_payload["burst_on_s"] = float(cfg.on_s)
    info_payload["burst_off_s"] = float(cfg.off_s)
    info_payload["burst_cycle_s"] = float(cycle)
    info_payload["burst_on_write_mult"] = float(cfg.on_write_mult)
    info_payload["burst_off_write_mult"] = float(cfg.off_write_mult)
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
        info_json=json.dumps(info_payload, sort_keys=True),
        measured_requests=result.measured_requests,
        measured_elapsed_s=result.measured_elapsed_s,
        warmup_requests=result.warmup_requests,
        warmup_elapsed_s=result.warmup_elapsed_s,
    )


def _parse_info_payload(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except Exception:
        return {"raw_info_json": payload_json}
    if not isinstance(payload, dict):
        return {"raw_info_json": payload}
    return dict(payload)
