"""Validate run artifacts for schema compliance."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import pandas as pd

from maxionbench.schemas.result_schema import REQUIRED_METADATA_FIELDS, read_metadata

REQUIRED_STAGE_TIMING_COLUMNS = (
    "setup_elapsed_s",
    "warmup_elapsed_s",
    "measure_elapsed_s",
    "export_elapsed_s",
)


def _is_run_directory(path: Path) -> bool:
    return (
        (path / "results.parquet").exists()
        and (path / "run_metadata.json").exists()
        and (path / "config_resolved.yaml").exists()
    )


def discover_run_directories(path: Path) -> list[Path]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    if _is_run_directory(resolved):
        return [resolved]
    run_dirs = sorted({candidate.parent.resolve() for candidate in resolved.rglob("results.parquet")})
    return run_dirs


def validate_run_directory(path: Path) -> dict[str, Any]:
    required_files = [
        path / "results.parquet",
        path / "run_metadata.json",
        path / "config_resolved.yaml",
    ]
    missing = [str(item) for item in required_files if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    frame = pd.read_parquet(path / "results.parquet")
    if frame.empty:
        raise ValueError("results.parquet must contain at least one row")
    missing_timing = [col for col in REQUIRED_STAGE_TIMING_COLUMNS if col not in frame.columns]
    if missing_timing:
        raise ValueError(f"results.parquet missing stage timing columns: {missing_timing}")
    for col in REQUIRED_STAGE_TIMING_COLUMNS:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"results.parquet column `{col}` contains non-numeric values")
        if (values < 0.0).any():
            raise ValueError(f"results.parquet column `{col}` contains negative values")

    metadata = read_metadata(path / "run_metadata.json")
    missing_keys = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing_keys:
        raise ValueError(f"run_metadata.json missing keys: {missing_keys}")

    return {
        "rows": int(len(frame)),
        "metadata_keys_ok": True,
        "stage_timing_ok": True,
    }


def validate_path(path: Path) -> dict[str, Any]:
    run_dirs = discover_run_directories(path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {path.resolve()}")
    runs: list[dict[str, Any]] = []
    total_rows = 0
    for run_dir in run_dirs:
        summary = validate_run_directory(run_dir)
        summary["run_path"] = str(run_dir)
        total_rows += int(summary["rows"])
        runs.append(summary)
    return {
        "input": str(path.resolve()),
        "run_count": len(runs),
        "total_rows": total_rows,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate MaxionBench outputs")
    parser.add_argument("--input", required=True, help="Run directory to validate")
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args(argv)
    summary = validate_path(Path(args.input).resolve())
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
