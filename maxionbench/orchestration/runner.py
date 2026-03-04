"""Runner entrypoint for MaxionBench benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from maxionbench.adapters import create_adapter
from maxionbench.datasets.d3_generator import D3Params, params_from_mapping
from maxionbench.datasets.loaders.d1_ann_hdf5 import D1AnnDataset, load_d1_ann_hdf5
from maxionbench.datasets.loaders.d2_bigann import D2BigAnnDataset, load_d2_bigann
from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset
from maxionbench.datasets.loaders.d4_text import load_d4_from_local_bundles
from maxionbench.metrics.cost_rhu import rhu_hours, rhu_rate
from maxionbench.orchestration.config_schema import RunConfig, load_run_config
from maxionbench.runtime.rpc_baseline import measure_rpc_baseline
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.scenarios.calibrate_d3 import CalibrateD3Config, run as run_calibrate_d3
from maxionbench.scenarios.matched_quality import MatchedQualityCandidate, select_candidate
from maxionbench.scenarios.s1_ann_frontier import S1Config, S1Data, run_with_data as run_s1_with_data
from maxionbench.scenarios.s2_filtered_ann import S2Config, run as run_s2
from maxionbench.scenarios.s3_churn_smooth import S3Config, run as run_s3
from maxionbench.scenarios.s3b_churn_bursty import S3bConfig, run as run_s3b
from maxionbench.scenarios.s4_hybrid import S4Config, run as run_s4
from maxionbench.scenarios.s5_rerank import S5Config, run as run_s5
from maxionbench.scenarios.s6_fusion import S6Config, run as run_s6
from maxionbench.schemas.result_schema import (
    ResultRow,
    RunMetadata,
    stable_config_fingerprint,
    utc_now_iso,
    write_resolved_config,
    write_results_parquet,
    write_run_metadata,
)


@dataclass(frozen=True)
class _SweepRun:
    client_count: int
    search_params: dict[str, Any]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    sla_violation_rate: float
    errors: int
    rhu_h: float
    rtt_baseline_ms_p50: float
    rtt_baseline_ms_p99: float
    warmup_target_s: float
    warmup_elapsed_s: float
    warmup_requests: int
    measure_target_s: float
    measure_elapsed_s: float
    measure_requests: int


@dataclass(frozen=True)
class _RagCandidate:
    label: str
    search_payload: dict[str, Any]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    sla_violation_rate: float
    errors: int
    rhu_h: float
    rtt_baseline_ms_p50: float
    rtt_baseline_ms_p99: float
    warmup_target_s: float
    warmup_elapsed_s: float
    warmup_requests: int
    measure_target_s: float
    measure_elapsed_s: float
    measure_requests: int


_RAG_NDCG_BANDS: list[tuple[str, float, float]] = [
    ("low", 0.00, 0.35),
    ("medium", 0.35, 0.55),
    ("high", 0.55, 1.0000001),
]


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Run a MaxionBench scenario.")
    parser.add_argument("--config", required=True, help="Path to scenario YAML config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--no-retry", action="store_true", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--d3-params", default=None, help="Optional path to d3_params.yaml")
    return parser.parse_args(argv)


def run_from_config(config_path: Path, cli_overrides: dict[str, Any] | None = None) -> Path:
    overrides = dict(cli_overrides or {})
    d3_params_path = overrides.pop("d3_params", None)
    cfg = load_run_config(config_path, overrides=overrides)
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    config_payload = cfg.as_dict()
    config_payload["d3_params"] = d3_params_path
    config_fingerprint = stable_config_fingerprint(config_payload)

    if cfg.scenario == "calibrate_d3":
        rows = _run_calibrate_rows(cfg=cfg, config_fingerprint=config_fingerprint, d3_params_path=d3_params_path)
    elif cfg.scenario == "s1_ann_frontier":
        rows = _run_s1_rows(cfg=cfg, config_fingerprint=config_fingerprint)
    elif cfg.scenario == "s2_filtered_ann":
        rows = _run_s2_rows(cfg=cfg, config_fingerprint=config_fingerprint, d3_params_path=d3_params_path)
    elif cfg.scenario == "s3_churn_smooth":
        rows = _run_s3_rows(cfg=cfg, config_fingerprint=config_fingerprint, d3_params_path=d3_params_path)
    elif cfg.scenario == "s3b_churn_bursty":
        rows = _run_s3b_rows(cfg=cfg, config_fingerprint=config_fingerprint, d3_params_path=d3_params_path)
    elif cfg.scenario == "s4_hybrid":
        rows = _run_s4_rows(cfg=cfg, config_fingerprint=config_fingerprint)
    elif cfg.scenario == "s5_rerank":
        rows = _run_s5_rows(cfg=cfg, config_fingerprint=config_fingerprint)
    elif cfg.scenario == "s6_fusion":
        rows = _run_s6_rows(cfg=cfg, config_fingerprint=config_fingerprint)
    else:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")

    if not rows:
        raise RuntimeError("No rows were produced.")
    write_results_parquet(output_dir / "results.parquet", rows)
    for row in rows:
        _append_log(output_dir / "logs" / "runner.log", row)

    first = rows[0]
    metadata = RunMetadata(
        run_id=first.run_id,
        timestamp_utc=utc_now_iso(),
        engine=cfg.engine,
        engine_version=cfg.engine_version,
        scenario=cfg.scenario,
        dataset_bundle=cfg.dataset_bundle,
        dataset_hash=cfg.dataset_hash,
        seed=cfg.seed,
        clients_read=cfg.clients_read,
        clients_write=cfg.clients_write,
        quality_target=cfg.quality_target,
        rtt_baseline_ms_p50=first.rtt_baseline_ms_p50,
        rtt_baseline_ms_p99=first.rtt_baseline_ms_p99,
        sla_threshold_ms=cfg.sla_threshold_ms,
        rhu_weights=asdict(cfg.weights),
        config_fingerprint=config_fingerprint,
        repeats=cfg.repeats,
        no_retry=cfg.no_retry,
        clients_read_grid=list(cfg.clients_grid),
        quality_targets=list(cfg.quality_targets),
        hardware_runtime=collect_system_info(),
    )
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    write_resolved_config(output_dir / "config_resolved.yaml", config_payload)
    return output_dir


def _run_calibrate_rows(*, cfg: RunConfig, config_fingerprint: str, d3_params_path: str | None) -> list[ResultRow]:
    params = _resolve_d3_params(cfg, d3_params_path)
    calibrate_cfg = CalibrateD3Config(
        vector_dim=cfg.vector_dim,
        num_vectors=cfg.num_vectors,
        seed=cfg.seed,
        output_params_path=cfg.output_d3_params_path,
        initial_params=params,
    )
    calibration = run_calibrate_d3(calibrate_cfg)
    eval_payload = {
        "test_a_median_concentration": calibration.eval.test_a_median_concentration,
        "test_b_cluster_spread": calibration.eval.test_b_cluster_spread,
        "p99_ratio_1pct_to_50pct": calibration.eval.p99_ratio_1pct_to_50pct,
        "recall_gap_50_minus_1": calibration.eval.recall_gap_50_minus_1,
        "trivial": calibration.eval.trivial,
        "iterations": calibration.iterations,
        "adjusted": calibration.adjusted,
    }
    row = ResultRow(
        run_id=_run_id(config_fingerprint, 0, cfg.clients_read, cfg.quality_target, suffix="calib"),
        timestamp_utc=utc_now_iso(),
        repeat_idx=0,
        engine=cfg.engine,
        engine_version=cfg.engine_version,
        scenario=cfg.scenario,
        dataset_bundle=cfg.dataset_bundle,
        dataset_hash=cfg.dataset_hash,
        seed=cfg.seed,
        clients_read=cfg.clients_read,
        clients_write=cfg.clients_write,
        quality_target=cfg.quality_target,
        search_params_json=json.dumps(eval_payload, sort_keys=True),
        recall_at_10=calibration.eval.recall_1pct,
        ndcg_at_10=calibration.eval.recall_50pct,
        mrr_at_10=calibration.eval.recall_gap_50_minus_1,
        p50_ms=calibration.eval.p99_50pct_ms,
        p95_ms=calibration.eval.p99_1pct_ms,
        p99_ms=calibration.eval.p99_1pct_ms,
        qps=0.0,
        rhu_h=0.0,
        sla_threshold_ms=cfg.sla_threshold_ms,
        sla_violation_rate=0.0,
        errors=0,
        rtt_baseline_ms_p50=0.0,
        rtt_baseline_ms_p99=0.0,
        warmup_target_s=cfg.warmup_s,
        warmup_elapsed_s=0.0,
        warmup_requests=0,
        measure_target_s=cfg.steady_state_s,
        measure_elapsed_s=0.0,
        measure_requests=0,
    )
    return [row]


def _run_s1_rows(*, cfg: RunConfig, config_fingerprint: str) -> list[ResultRow]:
    s1_data = _maybe_load_s1_data(cfg)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        for client_count in cfg.clients_grid:
            sweep_runs = _run_s1_sweep_for_client(
                cfg=cfg,
                repeat_idx=repeat_idx,
                client_count=client_count,
                s1_data=s1_data,
            )
            rows.extend(
                _select_matched_quality_rows(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    sweep_runs=sweep_runs,
                    config_fingerprint=config_fingerprint,
                )
            )
    return rows


def _run_s2_rows(*, cfg: RunConfig, config_fingerprint: str, d3_params_path: str | None) -> list[ResultRow]:
    d3_params = _resolve_d3_params(cfg, d3_params_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)

        scenario = run_s2(
            adapter=adapter,
            cfg=S2Config(
                vector_dim=cfg.vector_dim,
                num_vectors=cfg.num_vectors,
                num_queries=cfg.num_queries,
                top_k=cfg.top_k,
                clients_read=cfg.clients_read,
                sla_threshold_ms=cfg.sla_threshold_ms,
                selectivities=list(cfg.s2_selectivities),
                warmup_s=cfg.warmup_s,
                steady_state_s=cfg.steady_state_s,
                phase_timing_mode=cfg.phase_timing_mode,
                phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                search_params=cfg.search_sweep[0] if cfg.search_sweep else None,
            ),
            rng=np.random.default_rng(cfg.seed + repeat_idx),
            d3_params=d3_params,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        for cond in scenario:
            duration = max(cond.measured_elapsed_s, 1e-9)
            rows.append(
                ResultRow(
                    run_id=_run_id(
                        config_fingerprint,
                        repeat_idx,
                        cfg.clients_read,
                        cfg.quality_target,
                        suffix=f"s2_{_slug(cond.selectivity)}",
                    ),
                    timestamp_utc=utc_now_iso(),
                    repeat_idx=repeat_idx,
                    engine=cfg.engine,
                    engine_version=cfg.engine_version,
                    scenario=cfg.scenario,
                    dataset_bundle=cfg.dataset_bundle,
                    dataset_hash=cfg.dataset_hash,
                    seed=cfg.seed,
                    clients_read=cfg.clients_read,
                    clients_write=cfg.clients_write,
                    quality_target=cfg.quality_target,
                    search_params_json=json.dumps(
                        {
                            "selectivity": cond.selectivity,
                            "filter": json.loads(cond.filter_json),
                            "p99_inflation_vs_unfiltered": cond.p99_inflation_vs_unfiltered,
                        },
                        sort_keys=True,
                    ),
                    recall_at_10=cond.recall_at_10,
                    ndcg_at_10=cond.ndcg_at_10,
                    mrr_at_10=cond.mrr_at_10,
                    p50_ms=cond.p50_ms,
                    p95_ms=cond.p95_ms,
                    p99_ms=cond.p99_ms,
                    qps=cond.qps,
                    rhu_h=rhu_hours(duration_s=duration, rate=rate),
                    sla_threshold_ms=cfg.sla_threshold_ms,
                    sla_violation_rate=cond.sla_violation_rate,
                    errors=cond.errors,
                    rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                    rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                )
            )
    return rows


def _run_s3_rows(*, cfg: RunConfig, config_fingerprint: str, d3_params_path: str | None) -> list[ResultRow]:
    return _run_s3_like_rows(
        cfg=cfg,
        config_fingerprint=config_fingerprint,
        d3_params_path=d3_params_path,
        bursty=False,
    )


def _run_s3b_rows(*, cfg: RunConfig, config_fingerprint: str, d3_params_path: str | None) -> list[ResultRow]:
    return _run_s3_like_rows(
        cfg=cfg,
        config_fingerprint=config_fingerprint,
        d3_params_path=d3_params_path,
        bursty=True,
    )


def _run_s4_rows(*, cfg: RunConfig, config_fingerprint: str) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)

        scenario = run_s4(
            adapter=adapter,
            cfg=S4Config(
                vector_dim=cfg.vector_dim,
                num_vectors=cfg.num_vectors,
                num_queries=cfg.num_queries,
                top_k=cfg.top_k,
                clients_read=cfg.clients_read,
                sla_threshold_ms=cfg.sla_threshold_ms,
                warmup_s=cfg.warmup_s,
                steady_state_s=cfg.steady_state_s,
                phase_timing_mode=cfg.phase_timing_mode,
                phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                dense_candidates=cfg.s4_dense_candidates,
                bm25_candidates=cfg.s4_bm25_candidates,
                rrf_k=cfg.rrf_k,
                search_params=cfg.search_sweep[0] if cfg.search_sweep else None,
            ),
            rng=np.random.default_rng(cfg.seed + repeat_idx),
            dataset=d4_data,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        candidates: list[_RagCandidate] = []
        for cond in scenario:
            duration = max(cond.measured_elapsed_s, 1e-9)
            candidates.append(
                _RagCandidate(
                    label=cond.mode,
                    search_payload=json.loads(cond.info_json),
                    p50_ms=cond.p50_ms,
                    p95_ms=cond.p95_ms,
                    p99_ms=cond.p99_ms,
                    qps=cond.qps,
                    recall_at_10=cond.recall_at_10,
                    ndcg_at_10=cond.ndcg_at_10,
                    mrr_at_10=cond.mrr_at_10,
                    sla_violation_rate=cond.sla_violation_rate,
                    errors=cond.errors,
                    rhu_h=rhu_hours(duration_s=duration, rate=rate),
                    rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                    rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                )
            )
        rows.extend(
            _select_rag_band_rows(
                cfg=cfg,
                repeat_idx=repeat_idx,
                config_fingerprint=config_fingerprint,
                candidates=candidates,
                suffix_prefix="s4",
            )
        )
    return rows


def _run_s5_rows(*, cfg: RunConfig, config_fingerprint: str) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)

        scenario = run_s5(
            adapter=adapter,
            cfg=S5Config(
                vector_dim=cfg.vector_dim,
                num_vectors=cfg.num_vectors,
                num_queries=cfg.num_queries,
                top_k=cfg.top_k,
                clients_read=cfg.clients_read,
                sla_threshold_ms=cfg.sla_threshold_ms,
                candidate_budgets=list(cfg.s5_candidate_budgets),
                warmup_s=cfg.warmup_s,
                steady_state_s=cfg.steady_state_s,
                phase_timing_mode=cfg.phase_timing_mode,
                phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                reranker_model_id=cfg.s5_reranker_model_id,
                reranker_revision_tag=cfg.s5_reranker_revision_tag,
                reranker_max_seq_len=cfg.s5_reranker_max_seq_len,
                reranker_precision=cfg.s5_reranker_precision,
                reranker_batch_size=cfg.s5_reranker_batch_size,
                reranker_truncation=cfg.s5_reranker_truncation,
                search_params=cfg.search_sweep[0] if cfg.search_sweep else None,
            ),
            rng=np.random.default_rng(cfg.seed + repeat_idx),
            dataset=d4_data,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        candidates: list[_RagCandidate] = []
        for cond in scenario:
            duration = max(cond.measured_elapsed_s, 1e-9)
            payload = json.loads(cond.info_json)
            payload["delta_ndcg_at_10"] = cond.delta_ndcg_at_10
            candidates.append(
                _RagCandidate(
                    label=f"budget{cond.candidate_budget}",
                    search_payload=payload,
                    p50_ms=cond.p50_ms,
                    p95_ms=cond.p95_ms,
                    p99_ms=cond.p99_ms,
                    qps=cond.qps,
                    recall_at_10=cond.recall_at_10,
                    ndcg_at_10=cond.ndcg_at_10,
                    mrr_at_10=cond.mrr_at_10,
                    sla_violation_rate=cond.sla_violation_rate,
                    errors=cond.errors,
                    rhu_h=rhu_hours(duration_s=duration, rate=rate),
                    rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                    rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                )
            )
        rows.extend(
            _select_rag_band_rows(
                cfg=cfg,
                repeat_idx=repeat_idx,
                config_fingerprint=config_fingerprint,
                candidates=candidates,
                suffix_prefix="s5",
            )
        )
    return rows


def _run_s6_rows(*, cfg: RunConfig, config_fingerprint: str) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)

        scenario = run_s6(
            adapter=adapter,
            cfg=S6Config(
                vector_dim=cfg.vector_dim,
                num_vectors=cfg.num_vectors,
                num_queries=cfg.num_queries,
                top_k=cfg.top_k,
                clients_read=cfg.clients_read,
                sla_threshold_ms=cfg.sla_threshold_ms,
                warmup_s=cfg.warmup_s,
                steady_state_s=cfg.steady_state_s,
                phase_timing_mode=cfg.phase_timing_mode,
                phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                rrf_k=cfg.rrf_k,
                dense_a_candidates=cfg.s6_dense_a_candidates,
                dense_b_candidates=cfg.s6_dense_b_candidates,
                bm25_candidates=cfg.s6_bm25_candidates,
                search_params=cfg.search_sweep[0] if cfg.search_sweep else None,
            ),
            rng=np.random.default_rng(cfg.seed + repeat_idx),
            dataset=d4_data,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        candidates: list[_RagCandidate] = []
        for cond in scenario:
            duration = max(cond.measured_elapsed_s, 1e-9)
            candidates.append(
                _RagCandidate(
                    label=cond.mode,
                    search_payload=json.loads(cond.info_json),
                    p50_ms=cond.p50_ms,
                    p95_ms=cond.p95_ms,
                    p99_ms=cond.p99_ms,
                    qps=cond.qps,
                    recall_at_10=cond.recall_at_10,
                    ndcg_at_10=cond.ndcg_at_10,
                    mrr_at_10=cond.mrr_at_10,
                    sla_violation_rate=cond.sla_violation_rate,
                    errors=cond.errors,
                    rhu_h=rhu_hours(duration_s=duration, rate=rate),
                    rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                    rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                )
            )
        rows.extend(
            _select_rag_band_rows(
                cfg=cfg,
                repeat_idx=repeat_idx,
                config_fingerprint=config_fingerprint,
                candidates=candidates,
                suffix_prefix="s6",
            )
        )
    return rows


def _select_rag_band_rows(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    config_fingerprint: str,
    candidates: list[_RagCandidate],
    suffix_prefix: str,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for band_name, low, high in _RAG_NDCG_BANDS:
        feasible: list[_RagCandidate] = []
        for candidate in candidates:
            ndcg = candidate.ndcg_at_10
            if ndcg < low:
                continue
            if band_name != "high" and ndcg >= high:
                continue
            if band_name == "high" and ndcg > high:
                continue
            feasible.append(candidate)
        if not feasible:
            continue
        feasible.sort(key=lambda item: (item.rhu_h, item.p99_ms, -item.qps))
        selected = feasible[0]
        payload = dict(selected.search_payload)
        payload["rag_ndcg_band"] = band_name
        payload["rag_ndcg_range"] = [low, 1.0 if band_name == "high" else high]
        rows.append(
            ResultRow(
                run_id=_run_id(
                    config_fingerprint,
                    repeat_idx,
                    cfg.clients_read,
                    low,
                    suffix=f"{suffix_prefix}_{band_name}_{selected.label}",
                ),
                timestamp_utc=utc_now_iso(),
                repeat_idx=repeat_idx,
                engine=cfg.engine,
                engine_version=cfg.engine_version,
                scenario=cfg.scenario,
                dataset_bundle=cfg.dataset_bundle,
                dataset_hash=cfg.dataset_hash,
                seed=cfg.seed,
                clients_read=cfg.clients_read,
                clients_write=cfg.clients_write,
                quality_target=low,
                search_params_json=json.dumps(payload, sort_keys=True),
                recall_at_10=selected.recall_at_10,
                ndcg_at_10=selected.ndcg_at_10,
                mrr_at_10=selected.mrr_at_10,
                p50_ms=selected.p50_ms,
                p95_ms=selected.p95_ms,
                p99_ms=selected.p99_ms,
                qps=selected.qps,
                rhu_h=selected.rhu_h,
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=selected.sla_violation_rate,
                errors=selected.errors,
                rtt_baseline_ms_p50=selected.rtt_baseline_ms_p50,
                rtt_baseline_ms_p99=selected.rtt_baseline_ms_p99,
                warmup_target_s=selected.warmup_target_s,
                warmup_elapsed_s=selected.warmup_elapsed_s,
                warmup_requests=selected.warmup_requests,
                measure_target_s=selected.measure_target_s,
                measure_elapsed_s=selected.measure_elapsed_s,
                measure_requests=selected.measure_requests,
            )
        )
    return rows


def _run_s3_like_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    bursty: bool,
) -> list[ResultRow]:
    d3_params = _resolve_d3_params(cfg, d3_params_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)

        base_cfg = S3Config(
            vector_dim=cfg.vector_dim,
            num_vectors=cfg.num_vectors,
            num_queries=cfg.num_queries,
            top_k=cfg.top_k,
            sla_threshold_ms=cfg.sla_threshold_ms,
            warmup_s=cfg.warmup_s,
            steady_state_s=cfg.steady_state_s,
            phase_timing_mode=cfg.phase_timing_mode,
            lambda_req_s=cfg.lambda_req_s,
            read_rate=cfg.s3_read_rate,
            insert_rate=cfg.s3_insert_rate,
            update_rate=cfg.s3_update_rate,
            delete_rate=cfg.s3_delete_rate,
            maintenance_interval_s=cfg.maintenance_interval_s,
            max_events=cfg.s3_max_events,
        )
        if bursty:
            result = run_s3b(
                adapter=adapter,
                cfg=S3bConfig(
                    base=base_cfg,
                    on_s=cfg.s3b_on_s,
                    off_s=cfg.s3b_off_s,
                    on_write_mult=cfg.s3b_on_write_mult,
                    off_write_mult=cfg.s3b_off_write_mult,
                ),
                rng=np.random.default_rng(cfg.seed + repeat_idx),
                d3_params=d3_params,
            )
            suffix = "s3b"
        else:
            result = run_s3(
                adapter=adapter,
                cfg=base_cfg,
                rng=np.random.default_rng(cfg.seed + repeat_idx),
                d3_params=d3_params,
            )
            suffix = "s3"

        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read + cfg.clients_write)
        duration = max(result.measured_elapsed_s, 1e-9)
        rows.append(
            ResultRow(
                run_id=_run_id(config_fingerprint, repeat_idx, cfg.clients_read, cfg.quality_target, suffix=suffix),
                timestamp_utc=utc_now_iso(),
                repeat_idx=repeat_idx,
                engine=cfg.engine,
                engine_version=cfg.engine_version,
                scenario=cfg.scenario,
                dataset_bundle=cfg.dataset_bundle,
                dataset_hash=cfg.dataset_hash,
                seed=cfg.seed,
                clients_read=cfg.clients_read,
                clients_write=cfg.clients_write,
                quality_target=cfg.quality_target,
                search_params_json=result.info_json,
                recall_at_10=result.recall_at_10,
                ndcg_at_10=result.ndcg_at_10,
                mrr_at_10=result.mrr_at_10,
                p50_ms=result.p50_ms,
                p95_ms=result.p95_ms,
                p99_ms=result.p99_ms,
                qps=result.qps,
                rhu_h=rhu_hours(duration_s=duration, rate=rate),
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=result.sla_violation_rate,
                errors=result.errors,
                rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                warmup_target_s=cfg.warmup_s,
                warmup_elapsed_s=result.warmup_elapsed_s,
                warmup_requests=result.warmup_requests,
                measure_target_s=cfg.steady_state_s,
                measure_elapsed_s=result.measured_elapsed_s,
                measure_requests=result.measured_requests,
            )
        )
    return rows


def _run_id(config_fingerprint: str, repeat_idx: int, client_count: int, quality_target: float, suffix: str = "") -> str:
    base = f"run-{config_fingerprint[:10]}-r{repeat_idx}-c{client_count}-t{int(quality_target * 100)}"
    if suffix:
        return f"{base}-{suffix}"
    return base


def _run_s1_sweep_for_client(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    client_count: int,
    s1_data: S1Data | None,
) -> list[_SweepRun]:
    runs: list[_SweepRun] = []
    for search_params in cfg.search_sweep:
        candidate_rng = np.random.default_rng(cfg.seed + repeat_idx)
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")

        baseline = measure_rpc_baseline(request_fn=lambda: adapter.healthcheck(), request_count=cfg.rpc_baseline_requests)
        scenario_cfg = S1Config(
            vector_dim=cfg.vector_dim,
            num_vectors=cfg.num_vectors,
            num_queries=cfg.num_queries,
            top_k=cfg.top_k,
            clients_read=client_count,
            sla_threshold_ms=cfg.sla_threshold_ms,
            warmup_s=cfg.warmup_s,
            steady_state_s=cfg.steady_state_s,
            phase_timing_mode=cfg.phase_timing_mode,
            phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
            search_params=search_params,
        )
        result = run_s1_with_data(adapter=adapter, cfg=scenario_cfg, rng=candidate_rng, data=s1_data)
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")

        rate = _rhu_rate_for_cfg(cfg=cfg, stats=stats, client_count=client_count + cfg.clients_write)
        duration = max(result.measured_elapsed_s, 1e-9)
        runs.append(
            _SweepRun(
                client_count=client_count,
                search_params=dict(search_params),
                p50_ms=result.p50_ms,
                p95_ms=result.p95_ms,
                p99_ms=result.p99_ms,
                qps=result.qps,
                recall_at_10=result.recall_at_10,
                ndcg_at_10=result.ndcg_at_10,
                mrr_at_10=result.mrr_at_10,
                sla_violation_rate=result.sla_violation_rate,
                errors=result.errors,
                rhu_h=rhu_hours(duration_s=duration, rate=rate),
                rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                warmup_target_s=cfg.warmup_s,
                warmup_elapsed_s=result.warmup_elapsed_s,
                warmup_requests=result.warmup_requests,
                measure_target_s=cfg.steady_state_s,
                measure_elapsed_s=result.measured_elapsed_s,
                measure_requests=result.measured_requests,
            )
        )
    return runs


def _select_matched_quality_rows(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    client_count: int,
    sweep_runs: list[_SweepRun],
    config_fingerprint: str,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    candidates = [
        MatchedQualityCandidate(quality=r.recall_at_10, p99_ms=r.p99_ms, qps=r.qps, rhu_h=r.rhu_h, payload=r)
        for r in sweep_runs
    ]
    for target in cfg.quality_targets:
        selected = select_candidate(candidates, target_quality=target)
        if selected is None:
            continue
        run = selected.payload
        rows.append(
            ResultRow(
                run_id=_run_id(config_fingerprint, repeat_idx, client_count, target),
                timestamp_utc=utc_now_iso(),
                repeat_idx=repeat_idx,
                engine=cfg.engine,
                engine_version=cfg.engine_version,
                scenario=cfg.scenario,
                dataset_bundle=cfg.dataset_bundle,
                dataset_hash=cfg.dataset_hash,
                seed=cfg.seed,
                clients_read=client_count,
                clients_write=cfg.clients_write,
                quality_target=target,
                search_params_json=json.dumps(run.search_params, sort_keys=True),
                recall_at_10=run.recall_at_10,
                ndcg_at_10=run.ndcg_at_10,
                mrr_at_10=run.mrr_at_10,
                p50_ms=run.p50_ms,
                p95_ms=run.p95_ms,
                p99_ms=run.p99_ms,
                qps=run.qps,
                rhu_h=run.rhu_h,
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=run.sla_violation_rate,
                errors=run.errors,
                rtt_baseline_ms_p50=run.rtt_baseline_ms_p50,
                rtt_baseline_ms_p99=run.rtt_baseline_ms_p99,
                warmup_target_s=run.warmup_target_s,
                warmup_elapsed_s=run.warmup_elapsed_s,
                warmup_requests=run.warmup_requests,
                measure_target_s=run.measure_target_s,
                measure_elapsed_s=run.measure_elapsed_s,
                measure_requests=run.measure_requests,
            )
        )
    return rows


def _resolve_d3_params(cfg: RunConfig, d3_params_path: str | None) -> D3Params:
    if d3_params_path:
        with Path(d3_params_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"d3 params file must contain a mapping: {d3_params_path}")
        return params_from_mapping(payload, seed=cfg.d3_seed)
    return D3Params(
        k_clusters=cfg.d3_k_clusters,
        num_tenants=cfg.d3_num_tenants,
        num_acl_buckets=cfg.d3_num_acl_buckets,
        num_time_buckets=cfg.d3_num_time_buckets,
        beta_tenant=cfg.d3_beta_tenant,
        beta_acl=cfg.d3_beta_acl,
        beta_time=cfg.d3_beta_time,
        seed=cfg.d3_seed,
    )


def _rhu_rate_for_cfg(*, cfg: RunConfig, stats: Any, client_count: int) -> float:
    return rhu_rate(
        cpu_vcpu=max(1.0, float(client_count)),
        gpu_count=0.0,
        ram_gib=stats.ram_usage_bytes / (1024.0**3),
        disk_tb=stats.disk_usage_bytes / (1024.0**4),
        refs=cfg.references,
        weights=cfg.weights,
    )


def _maybe_load_s1_data(cfg: RunConfig) -> S1Data | None:
    if cfg.dataset_bundle != "D1":
        if cfg.dataset_bundle != "D2":
            return None
        if not cfg.d2_base_fvecs_path or not cfg.d2_query_fvecs_path:
            return None
        dataset_d2 = load_d2_bigann(
            base_fvecs=Path(cfg.d2_base_fvecs_path),
            query_fvecs=Path(cfg.d2_query_fvecs_path),
            gt_ivecs=Path(cfg.d2_gt_ivecs_path) if cfg.d2_gt_ivecs_path else None,
            max_vectors=cfg.num_vectors,
            max_queries=cfg.num_queries,
            top_k=max(cfg.top_k, 10),
        )
        return _to_s1_data_d2(dataset_d2)
    if not cfg.dataset_path:
        return None
    dataset = load_d1_ann_hdf5(
        Path(cfg.dataset_path),
        max_vectors=cfg.num_vectors,
        max_queries=cfg.num_queries,
        top_k=max(cfg.top_k, 10),
    )
    return _to_s1_data(dataset)


def _maybe_load_d4_data(cfg: RunConfig) -> D4RetrievalDataset | None:
    if cfg.dataset_bundle != "D4":
        return None
    if not cfg.d4_use_real_data:
        return None
    beir_root = Path(cfg.d4_beir_root) if cfg.d4_beir_root else None
    crag_path = Path(cfg.d4_crag_path) if cfg.d4_crag_path else None
    return load_d4_from_local_bundles(
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        beir_root=beir_root,
        beir_subsets=list(cfg.d4_beir_subsets),
        beir_split=cfg.d4_beir_split,
        crag_path=crag_path,
        include_crag=cfg.d4_include_crag,
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
    )


def _to_s1_data(dataset: D1AnnDataset) -> S1Data:
    return S1Data(
        ids=list(dataset.ids),
        vectors=np.asarray(dataset.vectors, dtype=np.float32),
        queries=np.asarray(dataset.queries, dtype=np.float32),
        ground_truth_ids=list(dataset.ground_truth_ids),
    )


def _to_s1_data_d2(dataset: D2BigAnnDataset) -> S1Data:
    return S1Data(
        ids=list(dataset.ids),
        vectors=np.asarray(dataset.vectors, dtype=np.float32),
        queries=np.asarray(dataset.queries, dtype=np.float32),
        ground_truth_ids=list(dataset.ground_truth_ids),
    )


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _append_log(path: Path, row: ResultRow) -> None:
    payload = {
        "run_id": row.run_id,
        "repeat_idx": row.repeat_idx,
        "scenario": row.scenario,
        "p99_ms": row.p99_ms,
        "qps": row.qps,
        "recall_at_10": row.recall_at_10,
        "errors": row.errors,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{payload}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "seed": args.seed,
        "repeats": args.repeats,
        "no_retry": args.no_retry if args.no_retry is True else None,
        "output_dir": args.output_dir,
        "d3_params": args.d3_params,
    }
    run_from_config(Path(args.config), overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
