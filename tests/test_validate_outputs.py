from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.runner import run_from_config
from maxionbench.tools.migrate_stage_timing import backfill_path
from maxionbench.tools.validate_outputs import validate_path, validate_run_directory


def _make_run(tmp_path: Path, *, name: str, seed: int, overrides: dict[str, object] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": seed,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / name),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 12,
        "num_vectors": 120,
        "num_queries": 8,
        "top_k": 10,
        "warmup_s": 0.02,
        "steady_state_s": 0.02,
        "phase_timing_mode": "strict",
        "phase_max_requests_per_phase": 12,
    }
    if overrides:
        config.update(overrides)
    cfg_path = tmp_path / f"{name}.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return run_from_config(cfg_path, cli_overrides=None)


def _make_pinned_s3_run_with_matched_s1_baseline(tmp_path: Path, *, seed: int = 101) -> Path:
    runs_root = tmp_path / "runs"
    _make_run(
        runs_root,
        name="run-s1-d3-baseline-protocol",
        seed=seed,
        overrides={
            "scenario": "s1_ann_frontier",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 0,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9, 0.95],
            "repeats": 3,
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 16,
        },
    )
    return _make_run(
        runs_root,
        name="run-s3-with-baseline-protocol",
        seed=seed + 1,
        overrides={
            "scenario": "s3_churn_smooth",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 8,
            "clients_grid": [32],
            "repeats": 3,
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "lambda_req_s": 1000.0,
            "s3_read_rate": 800.0,
            "s3_insert_rate": 100.0,
            "s3_update_rate": 50.0,
            "s3_delete_rate": 50.0,
            "maintenance_interval_s": 60.0,
            "sla_threshold_ms": 120.0,
            "s3_max_events": 60,
        },
    )


def _make_pinned_s2_run(tmp_path: Path, *, seed: int = 121) -> Path:
    runs_root = tmp_path / "runs_s2"
    return _make_run(
        runs_root,
        name="run-s2-protocol",
        seed=seed,
        overrides={
            "scenario": "s2_filtered_ann",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 0,
            "clients_grid": [32],
            "s2_selectivities": [0.001, 0.01, 0.1, 0.5],
            "repeats": 3,
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "sla_threshold_ms": 80.0,
        },
    )


def _make_pinned_s5_run(tmp_path: Path, *, seed: int = 151) -> Path:
    runs_root = tmp_path / "runs_s5"
    return _make_run(
        runs_root,
        name="run-s5-protocol",
        seed=seed,
        overrides={
            "scenario": "s5_rerank",
            "dataset_bundle": "D4",
            "dataset_hash": "synthetic-d4-v1",
            "clients_read": 16,
            "clients_write": 0,
            "clients_grid": [16],
            "s5_candidate_budgets": [50, 200, 1000],
            "s5_reranker_model_id": "BAAI/bge-reranker-base",
            "s5_reranker_revision_tag": "2026-03-04",
            "s5_reranker_max_seq_len": 512,
            "s5_reranker_precision": "fp16",
            "s5_reranker_batch_size": 32,
            "s5_reranker_truncation": "right",
            # Keep generation runnable in local tests; protocol checks can mutate this to true.
            "s5_require_hf_backend": False,
            "repeats": 3,
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "sla_threshold_ms": 300.0,
        },
    )


def test_validate_path_supports_parent_directory(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run1 = _make_run(runs_root, name="run-a", seed=17)
    run2 = _make_run(runs_root, name="run-b", seed=19)
    summary = validate_path(runs_root)
    assert {Path(item["run_path"]) for item in summary["runs"]} == {run1, run2}
    assert summary["strict_schema"] is True
    assert summary["enforce_protocol"] is False
    assert summary["warning_count"] == 0
    assert int(summary["run_count"]) == 2
    assert int(summary["total_rows"]) == 2


def test_migrate_stage_timing_backfills_legacy_results(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="legacy-run", seed=23)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["setup_elapsed_s", "warmup_elapsed_s", "measure_elapsed_s", "export_elapsed_s"])
    legacy.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="missing stage timing columns"):
        validate_run_directory(run_dir)

    dry_summary = backfill_path(run_dir, dry_run=True)
    assert int(dry_summary["changed_runs"]) == 1
    with pytest.raises(ValueError, match="missing stage timing columns"):
        validate_run_directory(run_dir)

    write_summary = backfill_path(run_dir, dry_run=False)
    assert int(write_summary["changed_runs"]) == 1
    validated = validate_run_directory(run_dir)
    assert validated["stage_timing_ok"] is True


def test_migrate_stage_timing_cli_smoke(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-cli", seed=29)
    code = cli_main(["migrate-stage-timing", "--input", str(run_dir), "--dry-run"])
    assert code == 0


def test_validate_run_directory_requires_runner_log(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-missing-log", seed=31)
    (run_dir / "logs" / "runner.log").unlink()

    with pytest.raises(FileNotFoundError, match="runner.log"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_malformed_runner_log(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-bad-log", seed=37)
    log_path = run_dir / "logs" / "runner.log"
    log_path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_run_directory(run_dir)

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    bad_event = {
        "timestamp_utc": "2026-03-04T00:00:00+00:00",
        "run_id": "bad-run-id",
        "config_fingerprint": metadata["config_fingerprint"],
        "repeat_idx": 0,
        "engine": "mock",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "p99_ms": 1.0,
        "qps": 1.0,
        "recall_at_10": 1.0,
        "sla_violation_rate": 0.0,
        "errors": 0,
    }
    log_path.write_text(json.dumps(bad_event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing run_id entries"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_missing_resource_columns(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-missing-resource-cols", seed=41)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["resource_cpu_vcpu", "resource_gpu_count", "resource_ram_gib", "resource_disk_tb", "rhu_rate"])
    legacy.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="missing resource columns"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_missing_resource_metadata(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-missing-resource-meta", seed=43)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("resource_profile", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required RHU metadata mapping: resource_profile"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_missing_ground_truth_metadata(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-missing-ground-truth-meta", seed=45)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("ground_truth_source", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing ground truth metadata keys"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_non_integer_ground_truth_k(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-bad-ground-truth-k", seed=46)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ground_truth_k"] = 10.5
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ground_truth_k"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_missing_hardware_runtime_metadata(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-missing-hardware-runtime-meta", seed=48)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("hardware_runtime", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hardware/runtime summary"):
        validate_run_directory(run_dir)


def test_validate_run_directory_rejects_invalid_config_checksum_field(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-bad-config-checksum", seed=49)
    config_path = run_dir / "config_resolved.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["dataset_path_sha256"] = "not-a-sha"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_path_sha256"):
        validate_run_directory(run_dir)


def test_validate_path_legacy_ok_allows_schema_drift_with_warnings(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-legacy-ok", seed=47)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["resource_cpu_vcpu", "resource_gpu_count", "resource_ram_gib", "resource_disk_tb", "rhu_rate"])
    legacy = legacy.drop(columns=["setup_elapsed_s", "warmup_elapsed_s", "measure_elapsed_s", "export_elapsed_s"])
    legacy.to_parquet(results_path, index=False)
    (run_dir / "logs" / "runner.log").unlink()

    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("resource_profile", None)
    metadata.pop("ground_truth_engine", None)
    metadata.pop("hardware_runtime", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = validate_path(run_dir, strict_schema=False)
    assert summary["strict_schema"] is False
    assert int(summary["warning_count"]) > 0
    run_summary = summary["runs"][0]
    assert run_summary["stage_timing_ok"] is False
    assert run_summary["resource_fields_ok"] is False
    assert run_summary["ground_truth_metadata_ok"] is False
    assert run_summary["hardware_runtime_ok"] is False
    assert run_summary["resource_metadata_ok"] is False
    assert run_summary["logs_ok"] is False


def test_validate_cli_legacy_ok_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _make_run(tmp_path, name="run-cli-legacy", seed=53)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["resource_cpu_vcpu", "resource_gpu_count", "resource_ram_gib", "resource_disk_tb", "rhu_rate"])
    legacy.to_parquet(results_path, index=False)

    code = cli_main(["validate", "--input", str(run_dir), "--legacy-ok", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strict_schema"] is False
    assert payload["enforce_protocol"] is False
    assert int(payload["warning_count"]) > 0


def test_validate_run_directory_enforce_protocol_rejects_smoke_runtime_values(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-protocol-fail", seed=59)
    with pytest.raises(ValueError, match="pinned protocol"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_path_enforce_protocol_legacy_ok_reports_warnings(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, name="run-protocol-legacy", seed=61)
    summary = validate_path(run_dir, strict_schema=False, enforce_protocol=True)
    assert summary["strict_schema"] is False
    assert summary["enforce_protocol"] is True
    assert int(summary["warning_count"]) > 0
    run_summary = summary["runs"][0]
    assert run_summary["runtime_protocol_ok"] is False


def test_validate_cli_legacy_ok_with_enforce_protocol_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path, name="run-cli-protocol-legacy", seed=67)
    code = cli_main(["validate", "--input", str(run_dir), "--legacy-ok", "--enforce-protocol", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strict_schema"] is False
    assert payload["enforce_protocol"] is True
    assert int(payload["warning_count"]) > 0


def test_validate_run_directory_enforce_protocol_rejects_scenario_pin_drift(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-scenario-drift",
        seed=71,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    with pytest.raises(ValueError, match="clients_grid"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_rejects_s1_quality_target_drift(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-quality-drift",
        seed=73,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    with pytest.raises(ValueError, match="quality_targets"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_accepts_s2_robustness_payloads(tmp_path: Path) -> None:
    s2_run = _make_pinned_s2_run(tmp_path, seed=127)
    summary = validate_run_directory(s2_run, enforce_protocol=True)
    assert summary["runtime_protocol_ok"] is True


def test_validate_run_directory_enforce_protocol_rejects_missing_s2_robustness_key(tmp_path: Path) -> None:
    s2_run = _make_pinned_s2_run(tmp_path, seed=131)
    results_path = s2_run / "results.parquet"
    frame = pd.read_parquet(results_path)
    payload = json.loads(str(frame.iloc[0]["search_params_json"]))
    assert isinstance(payload, dict)
    payload.pop("p99_inflation_vs_unfiltered", None)
    frame.loc[0, "search_params_json"] = json.dumps(payload, sort_keys=True)
    frame.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="p99_inflation_vs_unfiltered"):
        validate_run_directory(s2_run, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_rejects_s2_unfiltered_anchor_mismatch(tmp_path: Path) -> None:
    s2_run = _make_pinned_s2_run(tmp_path, seed=137)
    results_path = s2_run / "results.parquet"
    frame = pd.read_parquet(results_path)
    updated: list[str] = []
    for raw in frame["search_params_json"].tolist():
        payload = json.loads(str(raw))
        assert isinstance(payload, dict)
        if abs(float(payload.get("selectivity", 0.0)) - 1.0) < 1e-9:
            payload["p99_inflation_vs_unfiltered"] = 0.9
        updated.append(json.dumps(payload, sort_keys=True))
    frame["search_params_json"] = updated
    frame.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="unfiltered anchor"):
        validate_run_directory(s2_run, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_rejects_s5_non_hf_backend(tmp_path: Path) -> None:
    s5_run = _make_pinned_s5_run(tmp_path, seed=157)
    config_path = s5_run / "config_resolved.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    config_payload["s5_require_hf_backend"] = True
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="reranker.backend"):
        validate_run_directory(s5_run, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_accepts_s5_hf_backend_payloads(tmp_path: Path) -> None:
    s5_run = _make_pinned_s5_run(tmp_path, seed=163)
    config_path = s5_run / "config_resolved.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    config_payload["s5_require_hf_backend"] = True
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    results_path = s5_run / "results.parquet"
    frame = pd.read_parquet(results_path)
    updated: list[str] = []
    for raw in frame["search_params_json"].tolist():
        payload = json.loads(str(raw))
        assert isinstance(payload, dict)
        reranker = payload.get("reranker")
        assert isinstance(reranker, dict)
        reranker["backend"] = "hf_cross_encoder"
        reranker["device"] = "cuda"
        reranker["local_files_only"] = True
        reranker["runtime_errors"] = 0
        reranker["fallback_reason"] = None
        updated.append(json.dumps(payload, sort_keys=True))
    frame["search_params_json"] = updated
    frame.to_parquet(results_path, index=False)

    summary = validate_run_directory(s5_run, enforce_protocol=True)
    assert summary["runtime_protocol_ok"] is True


def test_validate_run_directory_enforce_protocol_rejects_non_paper_d3_params_file(tmp_path: Path) -> None:
    s2_run = _make_pinned_s2_run(tmp_path, seed=139)
    d3_params_path = tmp_path / "d3_params_bad.yaml"
    d3_params_payload = {
        "k_clusters": 4096,
        "num_tenants": 100,
        "num_acl_buckets": 16,
        "num_time_buckets": 52,
        "beta_tenant": 0.95,
        "beta_acl": 0.95,
        "beta_time": 0.90,
        "seed": 42,
        "calibration_eval": {
            "test_a_median_concentration": 0.547,
            "test_b_cluster_spread": 42.775,
            "p99_ratio_1pct_to_50pct": 1.19,
            "recall_gap_50_minus_1": -0.79,
            "trivial": True,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    d3_params_path.write_text(yaml.safe_dump(d3_params_payload, sort_keys=True), encoding="utf-8")

    config_path = s2_run / "config_resolved.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    config_payload["d3_params"] = str(d3_params_path)
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="not paper-ready"):
        validate_run_directory(s2_run, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_requires_gpu_tracks_omitted_flag(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-gpu-omission-field",
        seed=75,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9, 0.95],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("gpu_tracks_omitted", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_tracks_omitted"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_requires_pinned_rtt_baseline_request_profile(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-rtt-profile",
        seed=76,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9, 0.95],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["rtt_baseline_request_profile"] = "healthcheck_only"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rtt_baseline_request_profile"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_requires_dataset_cache_provenance_for_config_checksum(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-missing-cache-provenance",
        seed=77,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9, 0.95],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    config_path = run_dir / "config_resolved.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    config_payload["dataset_path"] = "/tmp/synthetic_d1_dataset.hdf5"
    config_payload["dataset_path_sha256"] = "a" * 64
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="missing provenance entry for dataset_path"):
        validate_run_directory(run_dir, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_accepts_dataset_cache_provenance_for_config_checksum(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(
        tmp_path,
        name="run-protocol-with-cache-provenance",
        seed=78,
        overrides={
            "repeats": 3,
            "clients_grid": [1, 8, 32, 64],
            "quality_targets": [0.8, 0.9, 0.95],
            "warmup_s": 120,
            "steady_state_s": 300,
            "rpc_baseline_requests": 1000,
            "phase_timing_mode": "bounded",
            "phase_max_requests_per_phase": 8,
        },
    )
    expected = "b" * 64
    config_path = run_dir / "config_resolved.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    config_payload["dataset_path"] = "/tmp/synthetic_d1_dataset.hdf5"
    config_payload["dataset_path_sha256"] = expected
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")

    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_cache_checksums"] = [
        {
            "path_key": "dataset_path",
            "resolved_path": "/tmp/synthetic_d1_dataset.hdf5",
            "source": "config key dataset_path_sha256",
            "expected_sha256": expected,
            "actual_sha256": expected,
        }
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = validate_run_directory(run_dir, enforce_protocol=True)
    assert summary["runtime_protocol_ok"] is True


def test_validate_path_enforce_protocol_flags_missing_s3_s1_baseline(tmp_path: Path) -> None:
    s3_run = _make_run(
        tmp_path,
        name="run-s3-only",
        seed=79,
        overrides={
            "scenario": "s3_churn_smooth",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 8,
            "clients_grid": [32],
            "sla_threshold_ms": 120.0,
            "s3_max_events": 60,
            "allow_missing_s3_baseline": True,
        },
    )
    summary = validate_path(s3_run, strict_schema=False, enforce_protocol=True)
    assert summary["cross_run_protocol_ok"] is False
    assert any("missing matched S1 baseline" in warning for warning in summary["warnings"])


def test_validate_path_enforce_protocol_accepts_s3_with_matching_s1_baseline(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_run(
        runs_root,
        name="run-s1-d3-baseline",
        seed=83,
        overrides={
            "scenario": "s1_ann_frontier",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 0,
            "clients_grid": [32],
            "quality_targets": [0.8],
            "sla_threshold_ms": 50.0,
        },
    )
    _make_run(
        runs_root,
        name="run-s3-with-baseline",
        seed=89,
        overrides={
            "scenario": "s3_churn_smooth",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 32,
            "clients_write": 8,
            "clients_grid": [32],
            "sla_threshold_ms": 120.0,
            "s3_max_events": 60,
        },
    )
    summary = validate_path(runs_root, strict_schema=False, enforce_protocol=True)
    assert summary["cross_run_protocol_ok"] is True
    assert not any("missing matched S1 baseline" in warning for warning in summary["warnings"])


def test_validate_run_directory_enforce_protocol_accepts_s3_baseline_provenance_payloads(tmp_path: Path) -> None:
    s3_run = _make_pinned_s3_run_with_matched_s1_baseline(tmp_path, seed=107)
    summary = validate_run_directory(s3_run, enforce_protocol=True)
    assert summary["runtime_protocol_ok"] is True


def test_validate_run_directory_enforce_protocol_rejects_missing_s3_baseline_provenance_keys(tmp_path: Path) -> None:
    s3_run = _make_pinned_s3_run_with_matched_s1_baseline(tmp_path, seed=113)
    results_path = s3_run / "results.parquet"
    frame = pd.read_parquet(results_path)
    mutated_payloads: list[str] = []
    for raw in frame["search_params_json"].tolist():
        payload = json.loads(str(raw))
        assert isinstance(payload, dict)
        payload.pop("s1_baseline_p99_ms", None)
        mutated_payloads.append(json.dumps(payload, sort_keys=True))
    frame["search_params_json"] = mutated_payloads
    frame.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="s1_baseline_p99_ms"):
        validate_run_directory(s3_run, enforce_protocol=True)


def test_validate_run_directory_enforce_protocol_rejects_s3_burst_clock_anchor_drift(tmp_path: Path) -> None:
    s3_run = _make_pinned_s3_run_with_matched_s1_baseline(tmp_path, seed=149)
    results_path = s3_run / "results.parquet"
    frame = pd.read_parquet(results_path)
    mutated_payloads: list[str] = []
    for raw in frame["search_params_json"].tolist():
        payload = json.loads(str(raw))
        assert isinstance(payload, dict)
        payload["burst_clock_anchor"] = "arrival_index"
        mutated_payloads.append(json.dumps(payload, sort_keys=True))
    frame["search_params_json"] = mutated_payloads
    frame.to_parquet(results_path, index=False)

    with pytest.raises(ValueError, match="burst_clock_anchor"):
        validate_run_directory(s3_run, enforce_protocol=True)
