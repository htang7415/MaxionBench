"""Runner entrypoint for MaxionBench benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from maxionbench.adapters import create_adapter
from maxionbench.datasets.cache_integrity import (
    load_dataset_manifest,
    resolve_expected_sha256_with_source,
    verify_file_sha256,
)
from maxionbench.datasets.d3_calibrate import PAPER_MIN_CALIBRATION_VECTORS, paper_calibration_issues
from maxionbench.datasets.d3_generator import D3Params, params_from_mapping
from maxionbench.datasets.loaders.d1_ann_hdf5 import D1AnnDataset, load_d1_ann_hdf5
from maxionbench.datasets.loaders.d2_bigann import D2BigAnnDataset, load_d2_bigann
from maxionbench.datasets.loaders.d3_vectors import load_d3_vectors
from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset
from maxionbench.datasets.loaders.d4_text import load_d4_from_local_bundles
from maxionbench.metrics.cost_rhu import rhu_hours
from maxionbench.metrics.robustness import p99_inflation
from maxionbench.metrics.resources import ResourceProfile, profile_from_adapter_stats, rhu_rate_for_profile
from maxionbench.orchestration.config_schema import RunConfig, load_run_config
from maxionbench.runtime.rpc_baseline import measure_rpc_baseline, minimal_rpc_request_fn
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
    PINNED_RTT_BASELINE_REQUEST_PROFILE,
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
    setup_elapsed_s: float
    warmup_target_s: float
    warmup_elapsed_s: float
    warmup_requests: int
    measure_target_s: float
    measure_elapsed_s: float
    measure_requests: int
    resource_cpu_vcpu: float
    resource_gpu_count: float
    resource_ram_gib: float
    resource_disk_tb: float
    rhu_rate: float


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
    setup_elapsed_s: float
    warmup_target_s: float
    warmup_elapsed_s: float
    warmup_requests: int
    measure_target_s: float
    measure_elapsed_s: float
    measure_requests: int
    resource_cpu_vcpu: float
    resource_gpu_count: float
    resource_ram_gib: float
    resource_disk_tb: float
    rhu_rate: float


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
    parser.add_argument("--enforce-readiness", action="store_true")
    parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    parser.add_argument("--allow-gpu-unavailable", action="store_true")
    return parser.parse_args(argv)


def run_from_config(config_path: Path, cli_overrides: dict[str, Any] | None = None) -> Path:
    overrides = dict(cli_overrides or {})
    resolved_config_path = config_path.resolve()
    d3_params_path = overrides.pop("d3_params", None)
    enforce_readiness = bool(overrides.pop("enforce_readiness", False))
    conformance_matrix = Path(str(overrides.pop("conformance_matrix", "artifacts/conformance/conformance_matrix.csv")))
    behavior_dir = Path(str(overrides.pop("behavior_dir", "docs/behavior")))
    allow_gpu_unavailable = bool(overrides.pop("allow_gpu_unavailable", False))
    cfg = load_run_config(config_path, overrides=overrides)
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    if enforce_readiness:
        from maxionbench.tools.pre_run_gate import evaluate_pre_run_gate

        gate_summary = evaluate_pre_run_gate(
            config_path=config_path.resolve(),
            conformance_matrix_path=conformance_matrix.resolve(),
            behavior_dir=behavior_dir.resolve(),
            allow_gpu_unavailable=allow_gpu_unavailable,
            allow_mock=True,
        )
        if not bool(gate_summary.get("pass", False)):
            raise RuntimeError(f"pre-run readiness gate failed: {json.dumps(gate_summary, sort_keys=True)}")

    config_payload = cfg.as_dict()
    dataset_cache_checksums = _collect_dataset_cache_checksum_provenance(
        cfg=cfg,
        config_path=resolved_config_path,
    )
    config_payload["d3_params"] = d3_params_path
    config_payload["readiness"] = {
        "enforced": enforce_readiness,
        "conformance_matrix": str(conformance_matrix),
        "behavior_dir": str(behavior_dir),
        "allow_gpu_unavailable": allow_gpu_unavailable,
    }
    config_fingerprint = stable_config_fingerprint(config_payload)

    if cfg.scenario == "calibrate_d3":
        rows = _run_calibrate_rows(
            cfg=cfg,
            config_fingerprint=config_fingerprint,
            d3_params_path=d3_params_path,
            config_path=resolved_config_path,
        )
    elif cfg.scenario == "s1_ann_frontier":
        rows = _run_s1_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
    elif cfg.scenario == "s2_filtered_ann":
        rows = _run_s2_rows(
            cfg=cfg,
            config_fingerprint=config_fingerprint,
            d3_params_path=d3_params_path,
            config_path=resolved_config_path,
        )
    elif cfg.scenario == "s3_churn_smooth":
        rows = _run_s3_rows(
            cfg=cfg,
            config_fingerprint=config_fingerprint,
            d3_params_path=d3_params_path,
            config_path=resolved_config_path,
        )
    elif cfg.scenario == "s3b_churn_bursty":
        rows = _run_s3b_rows(
            cfg=cfg,
            config_fingerprint=config_fingerprint,
            d3_params_path=d3_params_path,
            config_path=resolved_config_path,
        )
    elif cfg.scenario == "s4_hybrid":
        rows = _run_s4_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
    elif cfg.scenario == "s5_rerank":
        rows = _run_s5_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
    elif cfg.scenario == "s6_fusion":
        rows = _run_s6_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
    else:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")

    if not rows:
        raise RuntimeError("No rows were produced.")
    output_path = output_dir / "results.parquet"
    log_path = output_dir / "logs" / "runner.log"
    first = rows[0]
    ground_truth = _ground_truth_descriptor(cfg)
    hardware_runtime = collect_system_info()
    configured_gpu_omission = bool(config_payload.get("readiness", {}).get("allow_gpu_unavailable", False))
    observed_gpu_count = float(hardware_runtime.get("gpu_count", 0.0) or 0.0)
    gpu_tracks_omitted = configured_gpu_omission and observed_gpu_count <= 0.0
    gpu_tracks_omission_reason = (
        "GPU Track B/C omitted because allow_gpu_unavailable=true and observed gpu_count=0"
        if gpu_tracks_omitted
        else None
    )
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
        ground_truth_source=ground_truth["source"],
        ground_truth_metric=ground_truth["metric"],
        ground_truth_k=ground_truth["k"],
        ground_truth_engine=ground_truth["engine"],
        rtt_baseline_ms_p50=first.rtt_baseline_ms_p50,
        rtt_baseline_ms_p99=first.rtt_baseline_ms_p99,
        rtt_baseline_request_profile=PINNED_RTT_BASELINE_REQUEST_PROFILE,
        sla_threshold_ms=cfg.sla_threshold_ms,
        rhu_weights=asdict(cfg.weights),
        config_fingerprint=config_fingerprint,
        repeats=cfg.repeats,
        no_retry=cfg.no_retry,
        clients_read_grid=list(cfg.clients_grid),
        quality_targets=list(cfg.quality_targets),
        rhu_references=_rhu_references_payload(cfg),
        resource_profile=_summarize_resource_profile(rows),
        hardware_runtime=hardware_runtime,
        dataset_cache_checksums=dataset_cache_checksums,
        gpu_tracks_omitted=gpu_tracks_omitted,
        gpu_tracks_omission_reason=gpu_tracks_omission_reason,
    )
    export_start = time.perf_counter()
    write_results_parquet(output_path, rows)
    _write_runner_log(log_path, rows, config_fingerprint=config_fingerprint)
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    write_resolved_config(output_dir / "config_resolved.yaml", config_payload)
    export_elapsed_s = time.perf_counter() - export_start

    rows_with_export = [replace(row, export_elapsed_s=export_elapsed_s) for row in rows]
    write_results_parquet(output_path, rows_with_export)
    _write_runner_log(log_path, rows_with_export, config_fingerprint=config_fingerprint)
    return output_dir


def _run_calibrate_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    config_path: Path,
) -> list[ResultRow]:
    params = _resolve_d3_params(cfg, d3_params_path)
    resolved_dataset_path = _resolve_optional_config_value_path(value=cfg.dataset_path, config_path=config_path)
    calibrate_cfg = CalibrateD3Config(
        vector_dim=cfg.vector_dim,
        num_vectors=cfg.num_vectors,
        seed=cfg.seed,
        output_params_path=cfg.output_d3_params_path,
        initial_params=params,
        dataset_path=str(resolved_dataset_path) if resolved_dataset_path is not None else None,
        calibration_source="dataset_path" if cfg.dataset_path else "synthetic_vectors",
        calibration_dataset_hash=cfg.dataset_hash,
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
        setup_elapsed_s=0.0,
        warmup_target_s=cfg.warmup_s,
        warmup_elapsed_s=0.0,
        warmup_requests=0,
        measure_target_s=cfg.steady_state_s,
        measure_elapsed_s=0.0,
        measure_requests=0,
    )
    return [row]


def _run_s1_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    s1_data = _maybe_load_s1_data(cfg, config_path=config_path)
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


def _run_s2_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    config_path: Path,
) -> list[ResultRow]:
    d3_params = _resolve_d3_params(cfg, d3_params_path)
    d3_vectors = _maybe_load_d3_vectors(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start

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
            vectors=d3_vectors,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                    resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                    resource_gpu_count=resource_payload["gpu_count"],
                    resource_ram_gib=resource_payload["ram_gib"],
                    resource_disk_tb=resource_payload["disk_tb"],
                    rhu_rate=resource_payload["rhu_rate"],
                    sla_threshold_ms=cfg.sla_threshold_ms,
                    sla_violation_rate=cond.sla_violation_rate,
                    errors=cond.errors,
                    rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                    rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                    setup_elapsed_s=setup_elapsed_s,
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                )
            )
    return rows


def _run_s3_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    config_path: Path,
) -> list[ResultRow]:
    return _run_s3_like_rows(
        cfg=cfg,
        config_fingerprint=config_fingerprint,
        d3_params_path=d3_params_path,
        bursty=False,
        config_path=config_path,
    )


def _run_s3b_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    config_path: Path,
) -> list[ResultRow]:
    return _run_s3_like_rows(
        cfg=cfg,
        config_fingerprint=config_fingerprint,
        d3_params_path=d3_params_path,
        bursty=True,
        config_path=config_path,
    )


def _run_s4_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start

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
        profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                    setup_elapsed_s=setup_elapsed_s,
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                    resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                    resource_gpu_count=resource_payload["gpu_count"],
                    resource_ram_gib=resource_payload["ram_gib"],
                    resource_disk_tb=resource_payload["disk_tb"],
                    rhu_rate=resource_payload["rhu_rate"],
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


def _run_s5_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start

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
                require_hf_backend=cfg.s5_require_hf_backend,
                search_params=cfg.search_sweep[0] if cfg.search_sweep else None,
            ),
            rng=np.random.default_rng(cfg.seed + repeat_idx),
            dataset=d4_data,
        )
        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                    setup_elapsed_s=setup_elapsed_s,
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                    resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                    resource_gpu_count=resource_payload["gpu_count"],
                    resource_ram_gib=resource_payload["ram_gib"],
                    resource_disk_tb=resource_payload["disk_tb"],
                    rhu_rate=resource_payload["rhu_rate"],
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


def _run_s6_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    d4_data = _maybe_load_d4_data(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start

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
        profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=cfg.clients_read)
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                    setup_elapsed_s=setup_elapsed_s,
                    warmup_target_s=cfg.warmup_s,
                    warmup_elapsed_s=cond.warmup_elapsed_s,
                    warmup_requests=cond.warmup_requests,
                    measure_target_s=cfg.steady_state_s,
                    measure_elapsed_s=cond.measured_elapsed_s,
                    measure_requests=cond.measured_requests,
                    resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                    resource_gpu_count=resource_payload["gpu_count"],
                    resource_ram_gib=resource_payload["ram_gib"],
                    resource_disk_tb=resource_payload["disk_tb"],
                    rhu_rate=resource_payload["rhu_rate"],
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
                resource_cpu_vcpu=selected.resource_cpu_vcpu,
                resource_gpu_count=selected.resource_gpu_count,
                resource_ram_gib=selected.resource_ram_gib,
                resource_disk_tb=selected.resource_disk_tb,
                rhu_rate=selected.rhu_rate,
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=selected.sla_violation_rate,
                errors=selected.errors,
                rtt_baseline_ms_p50=selected.rtt_baseline_ms_p50,
                rtt_baseline_ms_p99=selected.rtt_baseline_ms_p99,
                setup_elapsed_s=selected.setup_elapsed_s,
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
    config_path: Path,
) -> list[ResultRow]:
    d3_params = _resolve_d3_params(cfg, d3_params_path)
    d3_vectors = _maybe_load_d3_vectors(cfg, config_path=config_path)
    baseline_missing = False
    baseline_error: str | None = None
    try:
        s1_baseline_p99_ms, baseline_match_rows, baseline_lookup_root = _resolve_s3_s1_baseline_p99(cfg=cfg)
    except RuntimeError as exc:
        if not cfg.allow_missing_s3_baseline:
            raise
        s1_baseline_p99_ms = None
        baseline_match_rows = 0
        baseline_lookup_root = str(Path(cfg.output_dir).resolve().parent)
        baseline_missing = True
        baseline_error = str(exc)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")
        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start

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
                vectors=d3_vectors,
            )
            suffix = "s3b"
        else:
            result = run_s3(
                adapter=adapter,
                cfg=base_cfg,
                rng=np.random.default_rng(cfg.seed + repeat_idx),
                d3_params=d3_params,
                vectors=d3_vectors,
            )
            suffix = "s3"
        info_payload = _parse_info_json(result.info_json)
        info_payload["s1_baseline_p99_ms"] = s1_baseline_p99_ms
        info_payload["s1_baseline_match_rows"] = baseline_match_rows
        info_payload["s1_baseline_lookup_root"] = baseline_lookup_root
        info_payload["s1_baseline_missing"] = baseline_missing
        if baseline_missing:
            info_payload["s1_baseline_error"] = baseline_error
            info_payload["p99_inflation_vs_s1_baseline"] = None
        else:
            assert s1_baseline_p99_ms is not None
            info_payload["p99_inflation_vs_s1_baseline"] = p99_inflation(result.p99_ms, s1_baseline_p99_ms)

        stats = adapter.stats()
        adapter.drop(collection="maxionbench")
        profile, rate = _resource_profile_and_rate_for_cfg(
            cfg=cfg,
            stats=stats,
            client_count=cfg.clients_read + cfg.clients_write,
        )
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                search_params_json=json.dumps(info_payload, sort_keys=True),
                recall_at_10=result.recall_at_10,
                ndcg_at_10=result.ndcg_at_10,
                mrr_at_10=result.mrr_at_10,
                p50_ms=result.p50_ms,
                p95_ms=result.p95_ms,
                p99_ms=result.p99_ms,
                qps=result.qps,
                rhu_h=rhu_hours(duration_s=duration, rate=rate),
                resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                resource_gpu_count=resource_payload["gpu_count"],
                resource_ram_gib=resource_payload["ram_gib"],
                resource_disk_tb=resource_payload["disk_tb"],
                rhu_rate=resource_payload["rhu_rate"],
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=result.sla_violation_rate,
                errors=result.errors,
                rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                setup_elapsed_s=setup_elapsed_s,
                warmup_target_s=cfg.warmup_s,
                warmup_elapsed_s=result.warmup_elapsed_s,
                warmup_requests=result.warmup_requests,
                measure_target_s=cfg.steady_state_s,
                measure_elapsed_s=result.measured_elapsed_s,
                measure_requests=result.measured_requests,
            )
        )
    return rows


def _parse_info_json(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except Exception:
        return {"raw_info_json": payload_json}
    if not isinstance(payload, dict):
        return {"raw_info_json": payload}
    return dict(payload)


def _resolve_s3_s1_baseline_p99(*, cfg: RunConfig) -> tuple[float, int, str]:
    lookup_root = Path(cfg.output_dir).resolve().parent
    p99_values: list[float] = []
    match_rows = 0
    for path in sorted(lookup_root.rglob("results.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        required = {"scenario", "engine", "dataset_bundle", "dataset_hash", "clients_read", "p99_ms"}
        if not required.issubset(frame.columns):
            continue
        mask = (
            (frame["scenario"] == "s1_ann_frontier")
            & (frame["engine"] == cfg.engine)
            & (frame["dataset_bundle"] == cfg.dataset_bundle)
            & (frame["dataset_hash"] == cfg.dataset_hash)
            & (frame["clients_read"] == cfg.clients_read)
        )
        if "clients_write" in frame.columns:
            mask = mask & (frame["clients_write"] == 0)
        matched = frame.loc[mask, "p99_ms"]
        if matched.empty:
            continue
        match_rows += int(len(matched))
        for value in matched.tolist():
            try:
                p99 = float(value)
            except (TypeError, ValueError):
                continue
            if p99 > 0:
                p99_values.append(p99)
    if not p99_values:
        raise RuntimeError(
            "S3/S3b requires a matched S1 baseline under the same run root. "
            f"Expected at least one s1_ann_frontier result with engine={cfg.engine!r}, "
            f"dataset_bundle={cfg.dataset_bundle!r}, dataset_hash={cfg.dataset_hash!r}, "
            f"clients_read={cfg.clients_read}. "
            f"Lookup root: {lookup_root}"
        )
    return float(np.median(np.asarray(p99_values, dtype=np.float64))), match_rows, str(lookup_root)


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
        setup_start = time.perf_counter()
        adapter = create_adapter(cfg.engine, **cfg.adapter_options)
        adapter.create(collection="maxionbench", dimension=cfg.vector_dim, metric="ip")

        baseline = measure_rpc_baseline(
            request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
            request_count=cfg.rpc_baseline_requests,
        )
        setup_elapsed_s = time.perf_counter() - setup_start
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

        profile, rate = _resource_profile_and_rate_for_cfg(
            cfg=cfg,
            stats=stats,
            client_count=client_count + cfg.clients_write,
        )
        resource_payload = _resource_payload(profile=profile, rate=rate)
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
                setup_elapsed_s=setup_elapsed_s,
                warmup_target_s=cfg.warmup_s,
                warmup_elapsed_s=result.warmup_elapsed_s,
                warmup_requests=result.warmup_requests,
                measure_target_s=cfg.steady_state_s,
                measure_elapsed_s=result.measured_elapsed_s,
                measure_requests=result.measured_requests,
                resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                resource_gpu_count=resource_payload["gpu_count"],
                resource_ram_gib=resource_payload["ram_gib"],
                resource_disk_tb=resource_payload["disk_tb"],
                rhu_rate=resource_payload["rhu_rate"],
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
                resource_cpu_vcpu=run.resource_cpu_vcpu,
                resource_gpu_count=run.resource_gpu_count,
                resource_ram_gib=run.resource_ram_gib,
                resource_disk_tb=run.resource_disk_tb,
                rhu_rate=run.rhu_rate,
                sla_threshold_ms=cfg.sla_threshold_ms,
                sla_violation_rate=run.sla_violation_rate,
                errors=run.errors,
                rtt_baseline_ms_p50=run.rtt_baseline_ms_p50,
                rtt_baseline_ms_p99=run.rtt_baseline_ms_p99,
                setup_elapsed_s=run.setup_elapsed_s,
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
        if _requires_paper_d3_calibration_check(cfg):
            min_vectors = max(1, int(getattr(cfg, "d3_min_calibration_vectors", PAPER_MIN_CALIBRATION_VECTORS)))
            issues = paper_calibration_issues(payload=payload, min_vectors=min_vectors)
            if issues and not bool(getattr(cfg, "allow_unverified_d3_params", False)):
                joined = "; ".join(issues[:4])
                raise ValueError(
                    "d3 params are not paper-ready for D3 robustness scenarios. "
                    f"Provided file: {Path(d3_params_path).resolve()}. "
                    "Re-run `calibrate_d3` on real D3 (LAION subset) scale and provide the regenerated params. "
                    f"Detected issues: {joined}"
                )
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


def _requires_paper_d3_calibration_check(cfg: RunConfig) -> bool:
    return (
        str(cfg.dataset_bundle).upper() == "D3"
        and str(cfg.scenario) in {"s2_filtered_ann", "s3_churn_smooth", "s3b_churn_bursty"}
    )


def _ground_truth_descriptor(cfg: RunConfig) -> dict[str, Any]:
    ann_k = max(1, min(10, int(cfg.top_k)))
    scenario = cfg.scenario.lower()
    dataset = cfg.dataset_bundle.upper()

    if scenario == "calibrate_d3":
        return {
            "source": "d3_calibration_eval",
            "metric": "calibration_proxy",
            "k": 0,
            "engine": "synthetic_calibration",
        }
    if dataset == "D4":
        if cfg.d4_use_real_data:
            return {
                "source": "official_beir_crag_qrels",
                "metric": "ndcg_at_10",
                "k": 10,
                "engine": "beir_crag_qrels",
            }
        return {
            "source": "synthetic_d4_qrels",
            "metric": "ndcg_at_10",
            "k": 10,
            "engine": "synthetic_qrels",
        }
    if dataset == "D3":
        if scenario == "s2_filtered_ann":
            source = "exact_filtered_subset"
        elif scenario in {"s3_churn_smooth", "s3b_churn_bursty"}:
            source = "exact_dynamic_topk"
        else:
            source = "exact_topk"
        return {
            "source": source,
            "metric": "recall_at_10",
            "k": ann_k,
            "engine": "numpy_exact",
        }
    if dataset == "D2" and cfg.d2_gt_ivecs_path:
        return {
            "source": "bigann_ivecs",
            "metric": "recall_at_10",
            "k": ann_k,
            "engine": "provided_ground_truth",
        }
    if dataset in {"D1", "D2"}:
        return {
            "source": "exact_topk",
            "metric": "recall_at_10",
            "k": ann_k,
            "engine": "numpy_exact",
        }
    return {
        "source": "unknown",
        "metric": "unknown",
        "k": ann_k,
        "engine": "unknown",
    }


def _resource_profile_and_rate_for_cfg(*, cfg: RunConfig, stats: Any, client_count: int) -> tuple[ResourceProfile, float]:
    profile = profile_from_adapter_stats(
        stats=stats,
        client_count=client_count,
        gpu_count=_gpu_count_for_cfg(cfg),
    )
    rate = rhu_rate_for_profile(profile=profile, refs=cfg.references, weights=cfg.weights)
    return profile, rate


def _resource_payload(*, profile: ResourceProfile, rate: float) -> dict[str, float]:
    return {
        "cpu_vcpu": float(profile.cpu_vcpu),
        "gpu_count": float(profile.gpu_count),
        "ram_gib": float(profile.ram_gib),
        "disk_tb": float(profile.disk_tb),
        "rhu_rate": float(rate),
    }


def _rhu_references_payload(cfg: RunConfig) -> dict[str, float]:
    refs = cfg.references
    return {
        "c_ref_vcpu": float(refs.c_ref_vcpu),
        "g_ref_gpu": float(refs.g_ref_gpu),
        "r_ref_gib": float(refs.r_ref_gib),
        "d_ref_tb": float(refs.d_ref_tb),
    }


def _summarize_resource_profile(rows: list[ResultRow]) -> dict[str, float]:
    if not rows:
        return {
            "cpu_vcpu": 0.0,
            "gpu_count": 0.0,
            "ram_gib": 0.0,
            "disk_tb": 0.0,
            "rhu_rate": 0.0,
        }
    return {
        "cpu_vcpu": float(np.median([row.resource_cpu_vcpu for row in rows])),
        "gpu_count": float(np.median([row.resource_gpu_count for row in rows])),
        "ram_gib": float(np.median([row.resource_ram_gib for row in rows])),
        "disk_tb": float(np.median([row.resource_disk_tb for row in rows])),
        "rhu_rate": float(np.median([row.rhu_rate for row in rows])),
    }


def _gpu_count_for_cfg(cfg: RunConfig) -> float:
    explicit = cfg.adapter_options.get("gpu_count")
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass
    normalized = cfg.engine.lower().replace("_", "-")
    if normalized == "faiss-gpu":
        return 1.0
    return 0.0


def _collect_dataset_cache_checksum_provenance(*, cfg: RunConfig, config_path: Path) -> list[dict[str, str]]:
    manifest = load_dataset_manifest(cfg.dataset_bundle)
    cfg_payload = cfg.as_dict()
    rows: list[dict[str, str]] = []
    checks: list[tuple[str, str, str, str, str]] = []
    if cfg.dataset_bundle == "D1":
        checks.append(
            (
                "dataset_path",
                "dataset_path_sha256",
                "cache_sha256_dataset_path",
                "D1 dataset_path",
                "dataset_path",
            )
        )
    elif cfg.dataset_bundle == "D2":
        checks.extend(
            [
                (
                    "d2_base_fvecs_path",
                    "d2_base_fvecs_sha256",
                    "cache_sha256_d2_base_fvecs_path",
                    "D2 d2_base_fvecs_path",
                    "d2_base_fvecs_path",
                ),
                (
                    "d2_query_fvecs_path",
                    "d2_query_fvecs_sha256",
                    "cache_sha256_d2_query_fvecs_path",
                    "D2 d2_query_fvecs_path",
                    "d2_query_fvecs_path",
                ),
                (
                    "d2_gt_ivecs_path",
                    "d2_gt_ivecs_sha256",
                    "cache_sha256_d2_gt_ivecs_path",
                    "D2 d2_gt_ivecs_path",
                    "d2_gt_ivecs_path",
                ),
            ]
        )
    elif cfg.dataset_bundle == "D3":
        checks.append(
            (
                "dataset_path",
                "dataset_path_sha256",
                "cache_sha256_dataset_path",
                "D3 dataset_path",
                "dataset_path",
            )
        )
    elif cfg.dataset_bundle == "D4" and cfg.d4_use_real_data and cfg.d4_include_crag:
        checks.append(
            (
                "d4_crag_path",
                "d4_crag_sha256",
                "cache_sha256_d4_crag_path",
                "D4 d4_crag_path",
                "d4_crag_path",
            )
        )

    for path_key, config_key, manifest_key, label, record_path_key in checks:
        raw_path = cfg_payload.get(path_key)
        expected, source = resolve_expected_sha256_with_source(
            config_payload=cfg_payload,
            manifest_payload=manifest,
            config_key=config_key,
            manifest_key=manifest_key,
            label=label,
        )
        if raw_path is None or raw_path == "":
            if expected is not None and isinstance(source, str) and source.startswith("config key "):
                raise ValueError(f"{label}: checksum provided but `{path_key}` is missing")
            continue
        if expected is None:
            continue
        resolved = _resolve_config_value_path(value=str(raw_path), config_path=config_path)
        actual = verify_file_sha256(path=resolved, expected_sha256=expected, label=label)
        rows.append(
            {
                "path_key": record_path_key,
                "resolved_path": str(resolved),
                "source": source or "",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return rows


def _resolve_config_value_path(*, value: str, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _resolve_optional_config_value_path(*, value: str | None, config_path: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _resolve_config_value_path(value=str(value), config_path=config_path)


def _maybe_load_s1_data(cfg: RunConfig, *, config_path: Path) -> S1Data | None:
    bundle = str(cfg.dataset_bundle).upper()
    if bundle != "D1":
        if bundle == "D2":
            base_fvecs = _resolve_optional_config_value_path(value=cfg.d2_base_fvecs_path, config_path=config_path)
            query_fvecs = _resolve_optional_config_value_path(value=cfg.d2_query_fvecs_path, config_path=config_path)
            gt_ivecs = _resolve_optional_config_value_path(value=cfg.d2_gt_ivecs_path, config_path=config_path)
            if base_fvecs is None or query_fvecs is None:
                return None
            dataset_d2 = load_d2_bigann(
                base_fvecs=base_fvecs,
                query_fvecs=query_fvecs,
                gt_ivecs=gt_ivecs,
                max_vectors=cfg.num_vectors,
                max_queries=cfg.num_queries,
                top_k=max(cfg.top_k, 10),
            )
            return _to_s1_data_d2(dataset_d2)
        if bundle == "D3":
            resolved_dataset_path = _resolve_optional_config_value_path(value=cfg.dataset_path, config_path=config_path)
            if resolved_dataset_path is None:
                return None
            expected_sha = str(cfg.dataset_path_sha256) if cfg.dataset_path_sha256 else None
            vectors = load_d3_vectors(
                resolved_dataset_path,
                max_vectors=cfg.num_vectors,
                expected_dim=cfg.vector_dim,
                expected_sha256=expected_sha,
            )
            ids = [f"doc-{idx:07d}" for idx in range(vectors.shape[0])]
            query_count = min(int(cfg.num_queries), int(vectors.shape[0]))
            query_idx = np.random.default_rng(cfg.seed).choice(vectors.shape[0], size=query_count, replace=False)
            queries = np.asarray(vectors[query_idx], dtype=np.float32)
            return S1Data(
                ids=ids,
                vectors=np.asarray(vectors, dtype=np.float32),
                queries=queries,
                ground_truth_ids=None,
            )
        return None
    if not cfg.dataset_path:
        return None
    resolved_dataset_path = _resolve_optional_config_value_path(value=cfg.dataset_path, config_path=config_path)
    if resolved_dataset_path is None:
        return None
    dataset = load_d1_ann_hdf5(
        resolved_dataset_path,
        max_vectors=cfg.num_vectors,
        max_queries=cfg.num_queries,
        top_k=max(cfg.top_k, 10),
    )
    return _to_s1_data(dataset)


def _maybe_load_d3_vectors(cfg: RunConfig, *, config_path: Path) -> np.ndarray | None:
    if cfg.dataset_bundle != "D3":
        return None
    resolved_dataset_path = _resolve_optional_config_value_path(value=cfg.dataset_path, config_path=config_path)
    if resolved_dataset_path is None:
        return None
    expected_sha = str(cfg.dataset_path_sha256) if cfg.dataset_path_sha256 else None
    return load_d3_vectors(
        resolved_dataset_path,
        max_vectors=cfg.num_vectors,
        expected_dim=cfg.vector_dim,
        expected_sha256=expected_sha,
    )


def _maybe_load_d4_data(cfg: RunConfig, *, config_path: Path) -> D4RetrievalDataset | None:
    if cfg.dataset_bundle != "D4":
        return None
    if not cfg.d4_use_real_data:
        return None
    beir_root = _resolve_optional_config_value_path(value=cfg.d4_beir_root, config_path=config_path)
    crag_path = _resolve_optional_config_value_path(value=cfg.d4_crag_path, config_path=config_path)
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


def _write_runner_log(path: Path, rows: list[ResultRow], *, config_fingerprint: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "timestamp_utc": row.timestamp_utc,
                "run_id": row.run_id,
                "config_fingerprint": config_fingerprint,
                "repeat_idx": row.repeat_idx,
                "engine": row.engine,
                "scenario": row.scenario,
                "dataset_bundle": row.dataset_bundle,
                "p99_ms": row.p99_ms,
                "qps": row.qps,
                "recall_at_10": row.recall_at_10,
                "sla_violation_rate": row.sla_violation_rate,
                "errors": row.errors,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "seed": args.seed,
        "repeats": args.repeats,
        "no_retry": args.no_retry if args.no_retry is True else None,
        "output_dir": args.output_dir,
        "d3_params": args.d3_params,
        "enforce_readiness": args.enforce_readiness,
        "conformance_matrix": args.conformance_matrix,
        "behavior_dir": args.behavior_dir,
        "allow_gpu_unavailable": args.allow_gpu_unavailable,
    }
    run_from_config(Path(args.config), overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
