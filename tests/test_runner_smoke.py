from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from maxionbench.orchestration.runner import run_from_config
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
    assert set(["recall_at_10", "p99_ms", "qps"]).issubset(frame.columns)

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
    assert int(row["warmup_requests"]) > 0
    assert int(row["measure_requests"]) > 0
    assert int(row["measure_requests"]) <= 12
    assert float(row["measure_elapsed_s"]) >= 0.0
