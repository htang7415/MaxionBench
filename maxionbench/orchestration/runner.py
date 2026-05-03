"""Runner entrypoint for MaxionBench benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from maxionbench.adapters import create_adapter
from maxionbench.datasets.cache_integrity import (
    load_dataset_manifest,
    resolve_expected_sha256_with_source,
    verify_file_sha256,
)
from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset
from maxionbench.datasets.loaders.d4_text import load_d4_from_local_bundles
from maxionbench.datasets.loaders.processed import (
    dataset_dir_sha256,
    load_processed_d4_bundle,
    load_processed_text_dataset,
)
from maxionbench.metrics.cost_rhu import rhu_hours
from maxionbench.metrics.robustness import p99_inflation
from maxionbench.metrics.resources import ResourceProfile, profile_from_adapter_stats, rhu_rate_for_profile
from maxionbench.orchestration.config_schema import RunConfig, load_run_config
from maxionbench.runtime.rpc_baseline import measure_rpc_baseline, minimal_rpc_request_fn
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.scenarios.portable_text_retrieval import PortableTextConfig, evaluate_text_queries, ingest_text_dataset
from maxionbench.scenarios.s2_streaming_memory import StreamingMemoryConfig, run as run_streaming_memory
from maxionbench.schemas.result_schema import (
    PINNED_RTT_BASELINE_REQUEST_PROFILE,
    ResultRow,
    RunMetadata,
    RunStatus,
    stable_config_fingerprint,
    utc_now_iso,
    write_resolved_config,
    write_results_parquet,
    write_run_metadata,
    write_run_status,
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
    error_examples: tuple[str, ...]
    resource_cpu_vcpu: float
    resource_gpu_count: float
    resource_ram_gib: float
    resource_disk_tb: float
    rhu_rate: float


@dataclass
class _PreparedS1Context:
    adapter: Any
    baseline: dict[str, float]
    prepared: Any
    stats: Any
    setup_elapsed_s: float


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


@dataclass(frozen=True)
class MatchedQualityCandidate:
    quality: float
    p99_ms: float
    qps: float
    rhu_h: float
    payload: Any


def select_candidate(
    candidates: list[MatchedQualityCandidate],
    *,
    quality_target: float | None = None,
    target_quality: float | None = None,
) -> MatchedQualityCandidate | None:
    threshold = float(quality_target if quality_target is not None else target_quality if target_quality is not None else 0.0)
    feasible = [candidate for candidate in candidates if candidate.quality >= threshold]
    if not feasible:
        return None
    return min(feasible, key=lambda candidate: (candidate.rhu_h, candidate.p99_ms, -candidate.qps))


_RAG_NDCG_BANDS: list[tuple[str, float, float]] = [
    ("low", 0.00, 0.35),
    ("medium", 0.35, 0.55),
    ("high", 0.55, 1.0000001),
]

_PORTABLE_BUDGETS: dict[str, dict[str, int]] = {
    "b0": {"warmup_s": 10, "steady_state_s": 10, "repeats": 1},
    "b1": {"warmup_s": 15, "steady_state_s": 30, "repeats": 1},
    "b2": {"warmup_s": 30, "steady_state_s": 60, "repeats": 2},
}


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Run a MaxionBench scenario.")
    parser.add_argument("--config", required=True, help="Path to scenario YAML config")
    parser.add_argument("--budget", default=None, help="MaxionBench budget level (b0, b1, or b2)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--no-retry", action="store_true", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enforce-readiness", action="store_true")
    parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    parser.add_argument("--allow-gpu-unavailable", action="store_true")
    return parser.parse_args(argv)


def run_from_config(config_path: Path, cli_overrides: dict[str, Any] | None = None) -> Path:
    overrides = dict(cli_overrides or {})
    resolved_config_path = config_path.resolve()
    enforce_readiness = bool(overrides.pop("enforce_readiness", False))
    conformance_matrix = Path(str(overrides.pop("conformance_matrix", "artifacts/conformance/conformance_matrix.csv")))
    behavior_dir = Path(str(overrides.pop("behavior_dir", "docs/behavior")))
    allow_gpu_unavailable = bool(overrides.pop("allow_gpu_unavailable", False))
    cfg = load_run_config(config_path, overrides=overrides)
    cfg = _apply_portable_budget(cfg=cfg, cli_overrides=cli_overrides or {})
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    try:
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
        config_payload["readiness"] = {
            "enforced": enforce_readiness,
            "conformance_matrix": str(conformance_matrix),
            "behavior_dir": str(behavior_dir),
            "allow_gpu_unavailable": allow_gpu_unavailable,
        }
        config_fingerprint = stable_config_fingerprint(config_payload)
        s1_sweep_diagnostics: list[dict[str, Any]] | None = None
        s1_selection_summary: list[dict[str, Any]] | None = None
        s1_diagnostics_path = output_dir / "logs" / "s1_sweep_diagnostics.jsonl"
        s1_selection_summary_path = output_dir / "logs" / "s1_selection_summary.json"

        if cfg.scenario == "s1_single_hop":
            rows = _run_portable_s1_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
        elif cfg.scenario == "s2_streaming_memory":
            rows = _run_portable_s2_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
        elif cfg.scenario == "s3_multi_hop":
            rows = _run_portable_s3_rows(cfg=cfg, config_fingerprint=config_fingerprint, config_path=resolved_config_path)
        else:
            raise ValueError(f"Unsupported scenario: {cfg.scenario}")

        if s1_sweep_diagnostics is not None:
            _write_jsonl(path=s1_diagnostics_path, payloads=s1_sweep_diagnostics)
        if s1_selection_summary is not None:
            _write_json(path=s1_selection_summary_path, payload=s1_selection_summary)
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
            "GPU-dependent workloads (Track B, Track C, S5) omitted because allow_gpu_unavailable=true and observed gpu_count=0"
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
            profile=cfg.profile,
            budget_level=cfg.budget_level,
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
            c_llm_in=cfg.c_llm_in,
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
        write_run_status(
            output_dir / "run_status.json",
            RunStatus(status="success", timestamp_utc=utc_now_iso(), exit_code=0),
        )
        return output_dir
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        try:
            write_run_status(
                output_dir / "run_status.json",
                RunStatus(status="failed", timestamp_utc=utc_now_iso(), detail=detail[:512]),
            )
        except Exception:
            pass
        raise


def _run_calibrate_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    d3_params_path: str | None,
    config_path: Path,
) -> list[ResultRow]:
    params = _resolve_d3_params(cfg, d3_params_path)
    processed_dataset_path = _resolve_optional_config_value_path(value=cfg.processed_dataset_path, config_path=config_path)
    if processed_dataset_path is not None and str(cfg.dataset_bundle).upper() == "D3":
        resolved_dataset_path = (processed_dataset_path / "base.npy").resolve()
        calibration_source = "processed_dataset_path"
    else:
        resolved_dataset_path = _resolve_optional_config_value_path(value=cfg.dataset_path, config_path=config_path)
        calibration_source = "dataset_path" if cfg.dataset_path else "synthetic_vectors"
    calibrate_cfg = CalibrateD3Config(
        vector_dim=cfg.vector_dim,
        num_vectors=cfg.num_vectors,
        seed=cfg.seed,
        output_params_path=cfg.output_d3_params_path,
        initial_params=params,
        dataset_path=str(resolved_dataset_path) if resolved_dataset_path is not None else None,
        require_real_data=bool(cfg.calibration_require_real_data),
        calibration_source=calibration_source,
        calibration_dataset_hash=cfg.dataset_hash,
    )
    calibration = run_calibrate_d3(calibrate_cfg)
    eval_payload = {
        "test_a_median_concentration": calibration.eval.test_a_median_concentration,
        "test_b_cluster_spread": calibration.eval.test_b_cluster_spread,
        "p99_ratio_1pct_to_50pct": calibration.eval.p99_ratio_1pct_to_50pct,
        "p99_ratio_50pct_to_1pct": calibration.eval.p99_ratio_50pct_to_1pct,
        "recall_gap_50_minus_1": calibration.eval.recall_gap_50_minus_1,
        "recall_gap_1_minus_50": calibration.eval.recall_gap_1_minus_50,
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
        mrr_at_10=calibration.eval.recall_gap_1_minus_50,
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


def _create_benchmark_adapter(*, cfg: RunConfig, metric: str = "ip") -> Any:
    adapter = create_adapter(cfg.engine, **cfg.adapter_options)
    if cfg.index_params:
        adapter.set_index_params(cfg.index_params)
    adapter.create(
        collection="maxionbench",
        dimension=cfg.vector_dim,
        metric=_normalize_benchmark_metric(metric),
    )
    return adapter


def _apply_portable_budget(*, cfg: RunConfig, cli_overrides: Mapping[str, Any]) -> RunConfig:
    if cfg.profile not in {"maxionbench", "portable-agentic"} and cfg.scenario not in {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}:
        return cfg
    budget = str(cfg.budget_level or "").strip().lower()
    if budget not in _PORTABLE_BUDGETS:
        return cfg
    budget_cfg = _PORTABLE_BUDGETS[budget]
    repeats = cfg.repeats if cli_overrides.get("repeats") is not None else int(budget_cfg["repeats"])
    return replace(
        cfg,
        warmup_s=int(budget_cfg["warmup_s"]),
        steady_state_s=int(budget_cfg["steady_state_s"]),
        repeats=repeats,
    )


def _run_s1_rows(
    *,
    cfg: RunConfig,
    config_fingerprint: str,
    config_path: Path,
) -> tuple[list[ResultRow], list[dict[str, Any]], list[dict[str, Any]]]:
    s1_data = _maybe_load_s1_data(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    sweep_diagnostics: list[dict[str, Any]] = []
    selection_summary: list[dict[str, Any]] = []
    prepared_contexts: dict[int, _PreparedS1Context] = {}
    try:
        for repeat_idx in range(cfg.repeats):
            prepared_context = _prepare_s1_context(
                cfg=cfg,
                repeat_idx=repeat_idx,
                s1_data=s1_data,
                prepared_contexts=prepared_contexts,
            )
            for client_count in cfg.clients_grid:
                sweep_runs = _run_s1_sweep_for_client(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    prepared_context=prepared_context,
                )
                sweep_diagnostics.extend(
                    _s1_sweep_diagnostics_payload(
                        cfg=cfg,
                        repeat_idx=repeat_idx,
                        client_count=client_count,
                        sweep_runs=sweep_runs,
                    )
                )
                selected_rows, client_selection_summary = _select_matched_quality_rows(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    sweep_runs=sweep_runs,
                    config_fingerprint=config_fingerprint,
                )
                rows.extend(selected_rows)
                selection_summary.extend(client_selection_summary)
    finally:
        for prepared_context in prepared_contexts.values():
            try:
                prepared_context.adapter.drop(collection="maxionbench")
            except Exception:
                pass
    return rows, sweep_diagnostics, selection_summary


def _run_portable_s1_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    dataset = _load_portable_s1_dataset(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        for client_count in cfg.clients_grid:
            sweep_runs: list[_SweepRun] = []
            for search_idx, search_params in enumerate(cfg.search_sweep):
                setup_start = time.perf_counter()
                adapter = _create_benchmark_adapter(cfg=cfg)
                baseline = measure_rpc_baseline(
                    request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
                    request_count=cfg.rpc_baseline_requests,
                )
                ingest_text_dataset(adapter, dataset)
                observations: list[dict[str, Any]] = []
                result = evaluate_text_queries(
                    adapter=adapter,
                    cfg=PortableTextConfig(
                        top_k=cfg.top_k,
                        clients_read=client_count,
                        sla_threshold_ms=cfg.sla_threshold_ms,
                        warmup_s=cfg.warmup_s,
                        steady_state_s=cfg.steady_state_s,
                        phase_timing_mode=cfg.phase_timing_mode,
                        phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                        search_params=search_params,
                    ),
                    dataset=dataset,
                    observation_sink=observations.append,
                )
                observation_path = _portable_observation_path(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    search_idx=search_idx,
                )
                _write_jsonl(path=observation_path, payloads=observations)
                stats = adapter.stats()
                adapter.drop(collection="maxionbench")
                profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=client_count)
                resource_payload = _resource_payload(profile=profile, rate=rate)
                duration = max(result.measured_elapsed_s, 1e-9)
                rhu_h = rhu_hours(duration_s=duration, rate=rate)
                setup_elapsed_s = time.perf_counter() - setup_start
                payload = _portable_payload(
                    cfg=cfg,
                    search_params=search_params,
                    primary_quality_metric="ndcg_at_10",
                    primary_quality_value=result.ndcg_at_10,
                    avg_retrieved_input_tokens=result.avg_retrieved_input_tokens,
                    measured_requests=result.measured_requests,
                    rhu_h=rhu_h,
                    extra={
                        "evidence_coverage_at_5": result.evidence_coverage_at_5,
                        "evidence_coverage_at_10": result.evidence_coverage_at_10,
                        "evidence_coverage_at_20": result.evidence_coverage_at_20,
                        "observation_path": str(observation_path),
                    },
                )
                sweep_runs.append(
                    _SweepRun(
                        client_count=client_count,
                        search_params=payload,
                        p50_ms=result.p50_ms,
                        p95_ms=result.p95_ms,
                        p99_ms=result.p99_ms,
                        qps=result.qps,
                        recall_at_10=result.recall_at_10,
                        ndcg_at_10=result.ndcg_at_10,
                        mrr_at_10=result.mrr_at_10,
                        sla_violation_rate=result.sla_violation_rate,
                        errors=result.errors,
                        rhu_h=rhu_h,
                        rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                        rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                        setup_elapsed_s=setup_elapsed_s,
                        warmup_target_s=cfg.warmup_s,
                        warmup_elapsed_s=result.warmup_elapsed_s,
                        warmup_requests=result.warmup_requests,
                        measure_target_s=cfg.steady_state_s,
                        measure_elapsed_s=result.measured_elapsed_s,
                        measure_requests=result.measured_requests,
                        error_examples=(),
                        resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                        resource_gpu_count=resource_payload["gpu_count"],
                        resource_ram_gib=resource_payload["ram_gib"],
                        resource_disk_tb=resource_payload["disk_tb"],
                        rhu_rate=resource_payload["rhu_rate"],
                    )
                )
            rows.extend(
                _select_portable_quality_rows(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    sweep_runs=sweep_runs,
                    config_fingerprint=config_fingerprint,
                    quality_getter=lambda run: run.ndcg_at_10,
                    suffix="portable_s1",
                )
            )
    return rows


def _run_portable_s2_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    background, events = _load_portable_s2_datasets(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    post_insert_floor = _portable_s2_post_insert_floor(cfg.budget_level)
    for repeat_idx in range(cfg.repeats):
        for client_count in cfg.clients_grid:
            sweep_runs: list[_SweepRun] = []
            for search_idx, search_params in enumerate(cfg.search_sweep):
                setup_start = time.perf_counter()
                adapter = _create_benchmark_adapter(cfg=cfg)
                baseline = measure_rpc_baseline(
                    request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
                    request_count=cfg.rpc_baseline_requests,
                )
                observations: list[dict[str, Any]] = []
                result = run_streaming_memory(
                    adapter=adapter,
                    cfg=StreamingMemoryConfig(
                        top_k=cfg.top_k,
                        clients_read=client_count,
                        clients_write=cfg.clients_write,
                        sla_threshold_ms=cfg.sla_threshold_ms,
                        warmup_s=cfg.warmup_s,
                        steady_state_s=cfg.steady_state_s,
                        phase_timing_mode=cfg.phase_timing_mode,
                        phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                        max_freshness_events=cfg.s2_max_freshness_events,
                        search_params=search_params,
                    ),
                    background=background,
                    events=events,
                    static_observation_sink=observations.append,
                    freshness_observation_sink=observations.append,
                )
                observation_path = _portable_observation_path(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    search_idx=search_idx,
                )
                _write_jsonl(path=observation_path, payloads=observations)
                stats = adapter.stats()
                adapter.drop(collection="maxionbench")
                profile, rate = _resource_profile_and_rate_for_cfg(
                    cfg=cfg,
                    stats=stats,
                    client_count=client_count + cfg.clients_write,
                )
                resource_payload = _resource_payload(profile=profile, rate=rate)
                duration = max(result.static.measured_elapsed_s, 1e-9)
                rhu_h = rhu_hours(duration_s=duration, rate=rate)
                setup_elapsed_s = time.perf_counter() - setup_start
                payload = _portable_payload(
                    cfg=cfg,
                    search_params=search_params,
                    primary_quality_metric="ndcg_at_10",
                    primary_quality_value=result.static.ndcg_at_10,
                    avg_retrieved_input_tokens=result.static.avg_retrieved_input_tokens,
                    measured_requests=result.static.measured_requests,
                    rhu_h=rhu_h,
                    extra={
                        "freshness_hit_at_1s": result.freshness_hit_at_1s,
                        "freshness_hit_at_5s": result.freshness_hit_at_5s,
                        "stale_answer_rate_at_5s": result.stale_answer_rate_at_5s,
                        "p95_visibility_latency_ms": result.p95_visibility_latency_ms,
                        "event_count": result.event_count,
                        "overlap_skipped_event_count": result.overlap_skipped_event_count,
                        "freshness_floor_for_budget": post_insert_floor,
                        "observation_path": str(observation_path),
                    },
                )
                sweep_runs.append(
                    _SweepRun(
                        client_count=client_count,
                        search_params=payload,
                        p50_ms=result.static.p50_ms,
                        p95_ms=result.static.p95_ms,
                        p99_ms=result.static.p99_ms,
                        qps=result.static.qps,
                        recall_at_10=result.static.recall_at_10,
                        ndcg_at_10=result.static.ndcg_at_10,
                        mrr_at_10=result.static.mrr_at_10,
                        sla_violation_rate=result.static.sla_violation_rate,
                        errors=result.static.errors,
                        rhu_h=rhu_h,
                        rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                        rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                        setup_elapsed_s=setup_elapsed_s,
                        warmup_target_s=cfg.warmup_s,
                        warmup_elapsed_s=result.static.warmup_elapsed_s,
                        warmup_requests=result.static.warmup_requests,
                        measure_target_s=cfg.steady_state_s,
                        measure_elapsed_s=result.static.measured_elapsed_s,
                        measure_requests=result.static.measured_requests,
                        error_examples=(),
                        resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                        resource_gpu_count=resource_payload["gpu_count"],
                        resource_ram_gib=resource_payload["ram_gib"],
                        resource_disk_tb=resource_payload["disk_tb"],
                        rhu_rate=resource_payload["rhu_rate"],
                    )
                )
            rows.extend(
                _select_portable_quality_rows(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    sweep_runs=[
                        run
                        for run in sweep_runs
                        if float(run.search_params.get("freshness_hit_at_5s", 0.0)) >= post_insert_floor
                    ],
                    config_fingerprint=config_fingerprint,
                    quality_getter=lambda run: run.ndcg_at_10,
                    suffix="portable_s2",
                )
            )
    return rows


def _run_portable_s3_rows(*, cfg: RunConfig, config_fingerprint: str, config_path: Path) -> list[ResultRow]:
    dataset = _load_portable_s3_dataset(cfg, config_path=config_path)
    rows: list[ResultRow] = []
    for repeat_idx in range(cfg.repeats):
        for client_count in cfg.clients_grid:
            sweep_runs: list[_SweepRun] = []
            for search_idx, search_params in enumerate(cfg.search_sweep):
                setup_start = time.perf_counter()
                adapter = _create_benchmark_adapter(cfg=cfg)
                baseline = measure_rpc_baseline(
                    request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
                    request_count=cfg.rpc_baseline_requests,
                )
                ingest_text_dataset(adapter, dataset)
                observations: list[dict[str, Any]] = []
                result = evaluate_text_queries(
                    adapter=adapter,
                    cfg=PortableTextConfig(
                        top_k=cfg.top_k,
                        clients_read=client_count,
                        sla_threshold_ms=cfg.sla_threshold_ms,
                        warmup_s=cfg.warmup_s,
                        steady_state_s=cfg.steady_state_s,
                        phase_timing_mode=cfg.phase_timing_mode,
                        phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
                        search_params=search_params,
                    ),
                    dataset=dataset,
                    observation_sink=observations.append,
                )
                observation_path = _portable_observation_path(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    search_idx=search_idx,
                )
                _write_jsonl(path=observation_path, payloads=observations)
                stats = adapter.stats()
                adapter.drop(collection="maxionbench")
                profile, rate = _resource_profile_and_rate_for_cfg(cfg=cfg, stats=stats, client_count=client_count)
                resource_payload = _resource_payload(profile=profile, rate=rate)
                duration = max(result.measured_elapsed_s, 1e-9)
                rhu_h = rhu_hours(duration_s=duration, rate=rate)
                setup_elapsed_s = time.perf_counter() - setup_start
                payload = _portable_payload(
                    cfg=cfg,
                    search_params=search_params,
                    primary_quality_metric="evidence_coverage@10",
                    primary_quality_value=result.evidence_coverage_at_10,
                    avg_retrieved_input_tokens=result.avg_retrieved_input_tokens,
                    measured_requests=result.measured_requests,
                    rhu_h=rhu_h,
                    extra={
                        "evidence_coverage_at_5": result.evidence_coverage_at_5,
                        "evidence_coverage_at_10": result.evidence_coverage_at_10,
                        "evidence_coverage_at_20": result.evidence_coverage_at_20,
                        "observation_path": str(observation_path),
                    },
                )
                sweep_runs.append(
                    _SweepRun(
                        client_count=client_count,
                        search_params=payload,
                        p50_ms=result.p50_ms,
                        p95_ms=result.p95_ms,
                        p99_ms=result.p99_ms,
                        qps=result.qps,
                        recall_at_10=result.recall_at_10,
                        ndcg_at_10=result.ndcg_at_10,
                        mrr_at_10=result.mrr_at_10,
                        sla_violation_rate=result.sla_violation_rate,
                        errors=result.errors,
                        rhu_h=rhu_h,
                        rtt_baseline_ms_p50=baseline["rtt_baseline_ms_p50"],
                        rtt_baseline_ms_p99=baseline["rtt_baseline_ms_p99"],
                        setup_elapsed_s=setup_elapsed_s,
                        warmup_target_s=cfg.warmup_s,
                        warmup_elapsed_s=result.warmup_elapsed_s,
                        warmup_requests=result.warmup_requests,
                        measure_target_s=cfg.steady_state_s,
                        measure_elapsed_s=result.measured_elapsed_s,
                        measure_requests=result.measured_requests,
                        error_examples=(),
                        resource_cpu_vcpu=resource_payload["cpu_vcpu"],
                        resource_gpu_count=resource_payload["gpu_count"],
                        resource_ram_gib=resource_payload["ram_gib"],
                        resource_disk_tb=resource_payload["disk_tb"],
                        rhu_rate=resource_payload["rhu_rate"],
                    )
                )
            rows.extend(
                _select_portable_quality_rows(
                    cfg=cfg,
                    repeat_idx=repeat_idx,
                    client_count=client_count,
                    sweep_runs=sweep_runs,
                    config_fingerprint=config_fingerprint,
                    quality_getter=lambda run: float(run.search_params.get("evidence_coverage_at_10", 0.0)),
                    suffix="portable_s3",
                )
            )
    return rows


def _prepare_s1_context(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    s1_data: S1Data | None,
    prepared_contexts: dict[int, _PreparedS1Context],
) -> _PreparedS1Context:
    cache_key = 0 if s1_data is not None else int(repeat_idx)
    cached = prepared_contexts.get(cache_key)
    if cached is not None:
        return cached

    candidate_rng = np.random.default_rng(cfg.seed + repeat_idx)
    setup_start = time.perf_counter()
    adapter = _create_benchmark_adapter(cfg=cfg, metric=s1_data.metric if s1_data is not None else "ip")
    baseline = measure_rpc_baseline(
        request_fn=minimal_rpc_request_fn(adapter=adapter, vector_dim=cfg.vector_dim),
        request_count=cfg.rpc_baseline_requests,
    )
    prepared = prepare_s1_with_data(
        adapter=adapter,
        cfg=S1Config(
            vector_dim=cfg.vector_dim,
            num_vectors=cfg.num_vectors,
            num_queries=cfg.num_queries,
            top_k=cfg.top_k,
            clients_read=1,
            sla_threshold_ms=cfg.sla_threshold_ms,
        ),
        rng=candidate_rng,
        data=s1_data,
    )
    prepared_context = _PreparedS1Context(
        adapter=adapter,
        baseline=baseline,
        prepared=prepared,
        stats=adapter.stats(),
        setup_elapsed_s=time.perf_counter() - setup_start,
    )
    prepared_contexts[cache_key] = prepared_context
    return prepared_context


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
        adapter = _create_benchmark_adapter(cfg=cfg)
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
        adapter = _create_benchmark_adapter(cfg=cfg)
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
        adapter = _create_benchmark_adapter(cfg=cfg)
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
        adapter = _create_benchmark_adapter(cfg=cfg)
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
    d3_dataset = _maybe_load_processed_d3_dataset(cfg, config_path=config_path)
    d3_vectors = None if d3_dataset is not None else _maybe_load_d3_vectors(cfg, config_path=config_path)
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
        adapter = _create_benchmark_adapter(cfg=cfg)
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
            clients_read=cfg.clients_read,
            clients_write=cfg.clients_write,
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
                dataset=d3_dataset,
            )
            suffix = "s3b"
        else:
            result = run_s3(
                adapter=adapter,
                cfg=base_cfg,
                rng=np.random.default_rng(cfg.seed + repeat_idx),
                d3_params=d3_params,
                vectors=d3_vectors,
                dataset=d3_dataset,
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


def _portable_observation_path(*, cfg: RunConfig, repeat_idx: int, client_count: int, search_idx: int) -> Path:
    safe_scenario = str(cfg.scenario).replace("/", "_")
    return (
        Path(cfg.output_dir).resolve()
        / "logs"
        / "observations"
        / f"{safe_scenario}_r{repeat_idx}_c{client_count}_s{search_idx}.jsonl"
    )


def _run_s1_sweep_for_client(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    client_count: int,
    prepared_context: _PreparedS1Context,
) -> list[_SweepRun]:
    runs: list[_SweepRun] = []
    for search_params in cfg.search_sweep:
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
        result = run_s1_prepared(
            adapter=prepared_context.adapter,
            cfg=scenario_cfg,
            prepared=prepared_context.prepared,
        )

        profile, rate = _resource_profile_and_rate_for_cfg(
            cfg=cfg,
            stats=prepared_context.stats,
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
                rtt_baseline_ms_p50=prepared_context.baseline["rtt_baseline_ms_p50"],
                rtt_baseline_ms_p99=prepared_context.baseline["rtt_baseline_ms_p99"],
                setup_elapsed_s=prepared_context.setup_elapsed_s,
                warmup_target_s=cfg.warmup_s,
                warmup_elapsed_s=result.warmup_elapsed_s,
                warmup_requests=result.warmup_requests,
                measure_target_s=cfg.steady_state_s,
                measure_elapsed_s=result.measured_elapsed_s,
                measure_requests=result.measured_requests,
                error_examples=result.error_examples,
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
) -> tuple[list[ResultRow], list[dict[str, Any]]]:
    rows: list[ResultRow] = []
    selection_summary: list[dict[str, Any]] = []
    candidates = [
        MatchedQualityCandidate(quality=r.recall_at_10, p99_ms=r.p99_ms, qps=r.qps, rhu_h=r.rhu_h, payload=r)
        for r in sweep_runs
    ]
    best_run = max(
        sweep_runs,
        key=lambda run: (run.recall_at_10, -run.errors, run.qps, -run.p99_ms),
    )
    first_error_examples = next((list(run.error_examples) for run in sweep_runs if run.error_examples), [])
    for target in cfg.quality_targets:
        selected = select_candidate(candidates, target_quality=target)
        summary = {
            "repeat_idx": repeat_idx,
            "client_count": client_count,
            "target_quality": target,
            "selected": selected is not None,
            "best_available_recall_at_10": best_run.recall_at_10,
            "best_available_search_params": dict(best_run.search_params),
            "best_available_errors": best_run.errors,
            "best_available_error_examples": first_error_examples,
        }
        if selected is None:
            selection_summary.append(summary)
            continue
        run = selected.payload
        summary.update(
            {
                "selected_search_params": dict(run.search_params),
                "selected_recall_at_10": run.recall_at_10,
                "selected_errors": run.errors,
            }
        )
        selection_summary.append(summary)
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
    return rows, selection_summary


def _select_portable_quality_rows(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    client_count: int,
    sweep_runs: list[_SweepRun],
    config_fingerprint: str,
    quality_getter: Any,
    suffix: str,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    candidates = [
        MatchedQualityCandidate(
            quality=float(quality_getter(run)),
            p99_ms=run.p99_ms,
            qps=run.qps,
            rhu_h=run.rhu_h,
            payload=run,
        )
        for run in sweep_runs
    ]
    for target in cfg.quality_targets:
        selected = select_candidate(candidates, target_quality=target)
        if selected is None:
            continue
        run = selected.payload
        rows.append(
            ResultRow(
                run_id=_run_id(config_fingerprint, repeat_idx, client_count, target, suffix=suffix),
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
                budget_level=_portable_row_string(run.search_params, "budget_level"),
                embedding_model=_portable_row_string(run.search_params, "embedding_model"),
                task_cost_est=_portable_row_float(run.search_params, "task_cost_est"),
                freshness_hit_at_1s=_portable_row_float(run.search_params, "freshness_hit_at_1s"),
                freshness_hit_at_5s=_portable_row_float(run.search_params, "freshness_hit_at_5s"),
                stale_answer_rate_at_5s=_portable_row_float(run.search_params, "stale_answer_rate_at_5s"),
                p95_visibility_latency_ms=_portable_row_float(run.search_params, "p95_visibility_latency_ms"),
                evidence_coverage_at_10=_portable_row_float(run.search_params, "evidence_coverage_at_10"),
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


def _portable_payload(
    *,
    cfg: RunConfig,
    search_params: Mapping[str, Any],
    primary_quality_metric: str,
    primary_quality_value: float,
    avg_retrieved_input_tokens: float,
    measured_requests: int,
    rhu_h: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_cost = float(rhu_h) / max(int(measured_requests), 1)
    embedding_cost = 0.0
    llm_context_cost = float(cfg.c_llm_in) * float(avg_retrieved_input_tokens)
    payload = {
        "profile": cfg.profile,
        "budget_level": cfg.budget_level,
        "embedding_model": cfg.embedding_model,
        "embedding_dim": cfg.embedding_dim,
        "search_params": dict(search_params),
        "primary_quality_metric": primary_quality_metric,
        "primary_quality_value": float(primary_quality_value),
        "avg_retrieved_input_tokens": float(avg_retrieved_input_tokens),
        "retrieval_cost_est": retrieval_cost,
        "embedding_cost_est": embedding_cost,
        "llm_context_cost_est": llm_context_cost,
        "task_cost_est": retrieval_cost + embedding_cost + llm_context_cost,
        "c_llm_in": float(cfg.c_llm_in),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _portable_s2_post_insert_floor(budget_level: str | None) -> float:
    normalized = str(budget_level or "").strip().lower()
    if normalized == "b1":
        return 0.6
    if normalized == "b2":
        return 0.8
    return 0.0


def _portable_row_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _portable_row_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _s1_sweep_diagnostics_payload(
    *,
    cfg: RunConfig,
    repeat_idx: int,
    client_count: int,
    sweep_runs: list[_SweepRun],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for run in sweep_runs:
        met_targets = [target for target in cfg.quality_targets if run.recall_at_10 >= target]
        payloads.append(
            {
                "repeat_idx": repeat_idx,
                "client_count": client_count,
                "search_params": dict(run.search_params),
                "recall_at_10": run.recall_at_10,
                "p99_ms": run.p99_ms,
                "qps": run.qps,
                "errors": run.errors,
                "error_examples": list(run.error_examples),
                "measure_requests": run.measure_requests,
                "measure_elapsed_s": run.measure_elapsed_s,
                "quality_targets_met": met_targets,
            }
        )
    return payloads


def _format_s1_no_rows_detail(
    *,
    cfg: RunConfig,
    selection_summary: list[dict[str, Any]],
    sweep_diagnostics: list[dict[str, Any]],
    diagnostics_path: Path,
) -> str:
    best_by_client: dict[int, float] = {}
    for item in selection_summary:
        client_count = int(item["client_count"])
        best_recall = float(item["best_available_recall_at_10"])
        current = best_by_client.get(client_count)
        if current is None or best_recall > current:
            best_by_client[client_count] = best_recall
    best_summary = ", ".join(
        f"c{client}={best_by_client[client]:.4f}"
        for client in sorted(best_by_client)
    )
    error_examples: list[str] = []
    for item in sweep_diagnostics:
        for example in item.get("error_examples", []):
            if example not in error_examples:
                error_examples.append(str(example))
            if len(error_examples) >= 2:
                break
        if len(error_examples) >= 2:
            break
    detail = (
        "No feasible matched-quality candidates for s1_ann_frontier. "
        f"Engine={cfg.engine}, dataset={cfg.dataset_bundle}, targets={list(cfg.quality_targets)}. "
        f"Best recall_at_10 by client: {best_summary or 'none'}. "
        f"Diagnostics: {diagnostics_path}."
    )
    if error_examples:
        detail += " Observed query errors: " + "; ".join(error_examples) + "."
    return detail


def _resolve_d3_params(cfg: RunConfig, d3_params_path: str | None) -> D3Params:
    requires_verified = _requires_verified_d3_params(cfg)
    if not d3_params_path and requires_verified:
        raise ValueError(
            "d3 params are required for strict D3 robustness scenarios. "
            "Run `calibrate_d3` first and pass the resulting d3_params.yaml via `--d3-params`."
        )
    if d3_params_path:
        with Path(d3_params_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"d3 params file must contain a mapping: {d3_params_path}")
        if _requires_paper_d3_calibration_check(cfg):
            min_vectors = max(1, int(getattr(cfg, "d3_min_calibration_vectors", PAPER_MIN_CALIBRATION_VECTORS)))
            issues = paper_calibration_issues(payload=payload, min_vectors=min_vectors)
            allow_unverified = bool(getattr(cfg, "allow_unverified_d3_params", False)) and not requires_verified
            if issues and not allow_unverified:
                joined = "; ".join(issues[:4])
                raise ValueError(
                    "d3 params are not paper-ready for D3 robustness scenarios. "
                    f"Provided file: {Path(d3_params_path).resolve()}. "
                    "Re-run `calibrate_d3` on real D3 (LAION subset) scale and provide the regenerated params. "
                    f"Detected issues: {joined}"
                )
        calibrated = params_from_mapping(payload, seed=cfg.d3_seed)
        return D3Params(
            k_clusters=cfg.d3_k_clusters,
            num_tenants=cfg.d3_num_tenants,
            num_acl_buckets=cfg.d3_num_acl_buckets,
            num_time_buckets=cfg.d3_num_time_buckets,
            beta_tenant=calibrated.beta_tenant,
            beta_acl=calibrated.beta_acl,
            beta_time=calibrated.beta_time,
            seed=calibrated.seed,
        )
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


def _requires_verified_d3_params(cfg: RunConfig) -> bool:
    return _requires_paper_d3_calibration_check(cfg) and str(cfg.phase_timing_mode).strip().lower() == "strict"


def _ground_truth_descriptor(cfg: RunConfig) -> dict[str, Any]:
    ann_k = max(1, min(10, int(cfg.top_k)))
    scenario = cfg.scenario.lower()
    dataset = cfg.dataset_bundle.upper()

    if scenario == "s1_single_hop":
        return {
            "source": "portable_text_qrels",
            "metric": "ndcg_at_10",
            "k": 10,
            "engine": "processed_text_qrels",
        }
    if scenario == "s2_streaming_memory":
        return {
            "source": "portable_text_qrels_plus_streaming_events",
            "metric": "ndcg_at_10",
            "k": 10,
            "engine": "processed_text_qrels",
        }
    if scenario == "s3_multi_hop":
        return {
            "source": "hotpot_portable_supporting_facts",
            "metric": "evidence_coverage@10",
            "k": 10,
            "engine": "hotpot_portable_qrels",
        }
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
    return 0.0


def _collect_dataset_cache_checksum_provenance(*, cfg: RunConfig, config_path: Path) -> list[dict[str, str]]:
    manifest = load_dataset_manifest(cfg.dataset_bundle)
    cfg_payload = cfg.as_dict()
    rows: list[dict[str, str]] = []
    checks: list[tuple[str, str, str, str, str]] = []
    raw_processed_path = cfg_payload.get("processed_dataset_path")
    expected_processed_sha = cfg_payload.get("processed_dataset_sha256")
    if raw_processed_path not in {None, ""} and expected_processed_sha not in {None, ""}:
        resolved_processed = _resolve_config_value_path(value=str(raw_processed_path), config_path=config_path)
        expected_text = str(expected_processed_sha).strip().lower()
        actual_processed_sha = dataset_dir_sha256(resolved_processed)
        if actual_processed_sha != expected_text:
            raise ValueError(
                "processed dataset sha256 mismatch for "
                f"{resolved_processed}: expected {expected_text}, got {actual_processed_sha}"
            )
        rows.append(
            {
                "path_key": "processed_dataset_path",
                "resolved_path": str(resolved_processed),
                "source": "config key processed_dataset_sha256",
                "expected_sha256": expected_text,
                "actual_sha256": actual_processed_sha,
            }
        )
    elif raw_processed_path in {None, ""} and expected_processed_sha not in {None, ""}:
        raise ValueError("processed_dataset_sha256 provided but `processed_dataset_path` is missing")
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
    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.exists():
        return config_relative
    repo_root = Path(__file__).resolve().parents[2]
    repo_relative = (repo_root / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    return config_relative


def _resolve_optional_config_value_path(*, value: str | None, config_path: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _resolve_config_value_path(value=str(value), config_path=config_path)


def _maybe_load_s1_data(cfg: RunConfig, *, config_path: Path) -> S1Data | None:
    bundle = str(cfg.dataset_bundle).upper()
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is not None:
        if bundle in {"D1", "D2"}:
            processed = load_processed_ann_dataset(
                processed_dataset_path,
                max_vectors=cfg.num_vectors,
                max_queries=cfg.num_queries,
                top_k=max(cfg.top_k, 10),
            )
            return S1Data(
                ids=processed.ids,
                vectors=np.asarray(processed.vectors, dtype=np.float32),
                queries=np.asarray(processed.queries, dtype=np.float32),
                ground_truth_ids=processed.ground_truth_ids,
                metric=_normalize_benchmark_metric(processed.metric),
            )
        if bundle == "D3":
            processed = load_processed_filtered_ann_dataset(
                processed_dataset_path,
                max_vectors=cfg.num_vectors,
                max_queries=cfg.num_queries,
                top_k=max(cfg.top_k, 10),
            )
            return S1Data(
                ids=processed.ids,
                vectors=np.asarray(processed.vectors, dtype=np.float32),
                queries=np.asarray(processed.queries, dtype=np.float32),
                ground_truth_ids=None,
                metric=_normalize_benchmark_metric(processed.metric),
            )
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
            ids = SequentialDocIdSequence(int(vectors.shape[0]))
            query_count = min(int(cfg.num_queries), int(vectors.shape[0]))
            query_idx = np.random.default_rng(cfg.seed).choice(vectors.shape[0], size=query_count, replace=False)
            queries = np.asarray(vectors[query_idx], dtype=np.float32)
            return S1Data(
                ids=ids,
                vectors=np.asarray(vectors, dtype=np.float32),
                queries=queries,
                ground_truth_ids=None,
                metric="ip",
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
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is not None:
        raise ValueError(
            "processed D3 datasets are not yet supported for S2 filtered execution: "
            "the current filtered workload code still expects generated correlated metadata rather than explicit per-query filters. "
            "Use dataset_path for the legacy path until the S2 D3 scenario migration is complete."
        )
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


def _maybe_load_processed_d3_dataset(cfg: RunConfig, *, config_path: Path) -> Any | None:
    if str(cfg.dataset_bundle).upper() != "D3":
        return None
    if str(cfg.scenario).strip().lower() not in {"s3_churn_smooth", "s3b_churn_bursty"}:
        return None
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is None:
        return None
    return load_processed_filtered_ann_dataset(
        processed_dataset_path,
        max_vectors=cfg.num_vectors,
        max_queries=cfg.num_queries,
        top_k=max(cfg.top_k, 10),
    )


def _maybe_load_d4_data(cfg: RunConfig, *, config_path: Path) -> D4RetrievalDataset | None:
    if cfg.dataset_bundle != "D4":
        return None
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is not None:
        return load_processed_d4_bundle(
            processed_dataset_path,
            vector_dim=cfg.vector_dim,
            seed=cfg.seed,
            beir_subsets=list(cfg.d4_beir_subsets),
            include_crag=cfg.d4_include_crag,
            max_docs=cfg.d4_max_docs,
            max_queries=cfg.d4_max_queries,
        )
    if not cfg.d4_use_real_data:
        return None
    beir_root = _resolve_optional_config_value_path(value=cfg.d4_beir_root, config_path=config_path)
    crag_path = _resolve_optional_config_value_path(value=cfg.d4_crag_path, config_path=config_path)
    crag_expected_sha256 = str(cfg.d4_crag_sha256) if cfg.d4_crag_sha256 else None
    return load_d4_from_local_bundles(
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        beir_root=beir_root,
        beir_subsets=list(cfg.d4_beir_subsets),
        beir_split=cfg.d4_beir_split,
        crag_path=crag_path,
        crag_expected_sha256=crag_expected_sha256,
        include_crag=cfg.d4_include_crag,
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
    )


def _load_portable_s1_dataset(cfg: RunConfig, *, config_path: Path) -> D4RetrievalDataset:
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is not None:
        return load_processed_d4_bundle(
            processed_dataset_path,
            vector_dim=cfg.vector_dim,
            seed=cfg.seed,
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
            require_precomputed_embeddings=bool(cfg.embedding_model),
            beir_subsets=list(cfg.d4_beir_subsets),
            include_crag=False,
            max_docs=cfg.d4_max_docs,
            max_queries=cfg.d4_max_queries,
        )
    beir_root = _resolve_optional_config_value_path(value=cfg.d4_beir_root, config_path=config_path)
    if beir_root is None:
        raise FileNotFoundError(
            "S1 requires processed_dataset_path or d4_beir_root pointing at local text bundles"
        )
    return load_d4_from_local_bundles(
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        beir_root=beir_root,
        beir_subsets=list(cfg.d4_beir_subsets),
        beir_split=cfg.d4_beir_split,
        crag_path=None,
        include_crag=False,
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
    )


def _load_portable_s2_datasets(cfg: RunConfig, *, config_path: Path) -> tuple[D4RetrievalDataset, D4RetrievalDataset]:
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is None:
        raise FileNotFoundError("S2 requires processed_dataset_path pointing at the processed D4 root")
    background = load_processed_d4_bundle(
        processed_dataset_path,
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        require_precomputed_embeddings=bool(cfg.embedding_model),
        beir_subsets=list(cfg.d4_beir_subsets),
        include_crag=False,
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
    )
    events_path = processed_dataset_path / "crag" / "small_slice"
    events = load_processed_text_dataset(
        events_path,
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        require_precomputed_embeddings=bool(cfg.embedding_model),
        # Preserve all event evidence docs before filling remaining capacity.
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
        prioritize_qrel_docs=True,
        min_query_retention_ratio=0.9,
    )
    return background, events


def _load_portable_s3_dataset(cfg: RunConfig, *, config_path: Path) -> D4RetrievalDataset:
    processed_dataset_path = _resolve_optional_config_value_path(
        value=cfg.processed_dataset_path,
        config_path=config_path,
    )
    if processed_dataset_path is None:
        raise FileNotFoundError("S3 requires processed_dataset_path pointing at the HotpotQA-MaxionBench dataset")
    return load_processed_text_dataset(
        processed_dataset_path,
        vector_dim=cfg.vector_dim,
        seed=cfg.seed,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        require_precomputed_embeddings=bool(cfg.embedding_model),
        max_docs=cfg.d4_max_docs,
        max_queries=cfg.d4_max_queries,
    )


def _to_s1_data(dataset: D1AnnDataset) -> S1Data:
    return S1Data(
        ids=dataset.ids,
        vectors=np.asarray(dataset.vectors, dtype=np.float32),
        queries=np.asarray(dataset.queries, dtype=np.float32),
        ground_truth_ids=dataset.ground_truth_ids,
        metric=_normalize_benchmark_metric(dataset.metric),
    )


def _to_s1_data_d2(dataset: D2BigAnnDataset) -> S1Data:
    return S1Data(
        ids=dataset.ids,
        vectors=np.asarray(dataset.vectors, dtype=np.float32),
        queries=np.asarray(dataset.queries, dtype=np.float32),
        ground_truth_ids=dataset.ground_truth_ids,
        metric=_normalize_benchmark_metric(dataset.metric),
    )


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _normalize_benchmark_metric(metric: str | None) -> str:
    normalized = str(metric or "ip").strip().lower()
    if normalized in {"ip", "inner_product", "dot"}:
        return "ip"
    if normalized in {"l2", "euclid", "euclidean"}:
        return "l2"
    if normalized in {"cos", "cosine", "angular"}:
        return "cos"
    raise ValueError(f"Unsupported benchmark metric: {metric}")


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "budget_level": args.budget,
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
