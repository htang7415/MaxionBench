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
