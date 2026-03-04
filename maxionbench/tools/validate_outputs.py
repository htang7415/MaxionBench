"""Validate run artifacts for schema compliance."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd

from maxionbench.schemas.result_schema import REQUIRED_METADATA_FIELDS, read_metadata


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

    metadata = read_metadata(path / "run_metadata.json")
    missing_keys = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing_keys:
        raise ValueError(f"run_metadata.json missing keys: {missing_keys}")

    return {
        "rows": int(len(frame)),
        "metadata_keys_ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate MaxionBench outputs")
    parser.add_argument("--input", required=True, help="Run directory to validate")
    args = parser.parse_args(argv)
    validate_run_directory(Path(args.input).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
