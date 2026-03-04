from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.runner import run_from_config
from maxionbench.tools.migrate_stage_timing import backfill_path
from maxionbench.tools.validate_outputs import validate_path, validate_run_directory


def _make_run(tmp_path: Path, *, name: str, seed: int) -> Path:
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
