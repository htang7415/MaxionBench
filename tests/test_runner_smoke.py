from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from maxionbench.orchestration.runner import run_from_config
from maxionbench.orchestration.config_schema import RunConfig
from maxionbench.orchestration.runner import _gpu_count_for_cfg
from maxionbench.tools.validate_outputs import validate_run_directory


def test_runner_end_to_end(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 7,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 20,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 200,
        "num_queries": 20,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "config_resolved.yaml").exists()
    assert (out_dir / "logs" / "runner.log").exists()

    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    assert set(
        [
            "recall_at_10",
            "p99_ms",
            "qps",
            "resource_cpu_vcpu",
            "resource_gpu_count",
            "resource_ram_gib",
            "resource_disk_tb",
            "rhu_rate",
        ]
    ).issubset(frame.columns)
    row = frame.iloc[0]
    assert float(row["resource_cpu_vcpu"]) >= 1.0
    assert float(row["resource_ram_gib"]) >= 0.0
    assert float(row["resource_disk_tb"]) >= 0.0
    assert float(row["rhu_rate"]) > 0.0

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["resource_profile"].keys()) == {"cpu_vcpu", "gpu_count", "ram_gib", "disk_tb", "rhu_rate"}
    assert set(metadata["rhu_references"].keys()) == {"c_ref_vcpu", "g_ref_gpu", "r_ref_gib", "d_ref_tb"}
    log_lines = [line for line in (out_dir / "logs" / "runner.log").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(log_lines) == len(frame)
    event = json.loads(log_lines[0])
    assert event["run_id"] == row["run_id"]
    assert event["config_fingerprint"] == metadata["config_fingerprint"]
    assert event["scenario"] == row["scenario"]
    assert event["engine"] == row["engine"]

    summary = validate_run_directory(out_dir)
    assert summary["rows"] == 1


def test_runner_matched_quality_grid_outputs_one_row_per_target_and_client(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 11,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-grid"),
        "quality_target": 0.8,
        "quality_targets": [0.5, 0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1, 2],
        "search_sweep": [{"hnsw_ef": 32}, {"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 150,
        "num_queries": 10,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config-grid.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 4
    assert set(frame["quality_target"].tolist()) == {0.5, 0.8}
    assert set(frame["clients_read"].tolist()) == {1, 2}


def test_runner_phase_fields_and_strict_mode_cap(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 31,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-phase"),
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
    cfg_path = tmp_path / "config-phase.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert float(row["setup_elapsed_s"]) >= 0.0
    assert int(row["warmup_requests"]) > 0
    assert int(row["measure_requests"]) > 0
    assert int(row["measure_requests"]) <= 12
    assert float(row["measure_elapsed_s"]) >= 0.0
    assert float(row["export_elapsed_s"]) >= 0.0


def test_validate_outputs_rejects_missing_or_negative_stage_timing(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 43,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-validate"),
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
    cfg_path = tmp_path / "config-validate.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")

    missing_col = frame.drop(columns=["setup_elapsed_s"])
    missing_col.to_parquet(out_dir / "results.parquet", index=False)
    with pytest.raises(ValueError, match="missing stage timing columns"):
        validate_run_directory(out_dir)

    frame.loc[:, "export_elapsed_s"] = -1.0
    frame.to_parquet(out_dir / "results.parquet", index=False)
    with pytest.raises(ValueError, match="negative values"):
        validate_run_directory(out_dir)


def test_gpu_count_resolution() -> None:
    faiss_gpu_cfg = RunConfig(engine="faiss-gpu", no_retry=True)
    assert _gpu_count_for_cfg(faiss_gpu_cfg) == 1.0

    explicit_cfg = RunConfig(engine="mock", adapter_options={"gpu_count": 2}, no_retry=True)
    assert _gpu_count_for_cfg(explicit_cfg) == 2.0
