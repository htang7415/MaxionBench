"""Backfill legacy run artifacts with stage timing columns."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from maxionbench.tools.validate_outputs import REQUIRED_STAGE_TIMING_COLUMNS, discover_run_directories

DEFAULT_STAGE_TIMING_VALUES: Mapping[str, float] = {
    "setup_elapsed_s": 0.0,
    "warmup_elapsed_s": 0.0,
    "measure_elapsed_s": 0.0,
    "export_elapsed_s": 0.0,
}


def backfill_run_directory(path: Path, *, dry_run: bool) -> dict[str, Any]:
    results_path = path / "results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.parquet: {results_path}")

    frame = pd.read_parquet(results_path)
    if frame.empty:
        raise ValueError(f"results.parquet must contain at least one row: {results_path}")

    added_columns: list[str] = []
    for column in REQUIRED_STAGE_TIMING_COLUMNS:
        if column in frame.columns:
            continue
        default_value = float(DEFAULT_STAGE_TIMING_VALUES[column])
        frame[column] = default_value
        added_columns.append(column)

    changed = bool(added_columns)
    if changed and not dry_run:
        frame.to_parquet(results_path, index=False)

    return {
        "run_path": str(path.resolve()),
        "rows": int(len(frame)),
        "changed": changed,
        "added_columns": added_columns,
    }


def backfill_path(path: Path, *, dry_run: bool) -> dict[str, Any]:
    run_dirs = discover_run_directories(path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {path.resolve()}")

    runs: list[dict[str, Any]] = []
    changed_runs = 0
    for run_dir in run_dirs:
        summary = backfill_run_directory(run_dir, dry_run=dry_run)
        if summary["changed"]:
            changed_runs += 1
        runs.append(summary)

    return {
        "input": str(path.resolve()),
        "dry_run": dry_run,
        "run_count": len(runs),
        "changed_runs": changed_runs,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Backfill stage timing columns in legacy run artifacts")
    parser.add_argument("--input", required=True, help="Run directory or parent directory containing runs")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args(argv)

    summary = backfill_path(Path(args.input).resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
