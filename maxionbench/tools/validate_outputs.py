"""Validate run artifacts for schema compliance."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from maxionbench.schemas.result_schema import REQUIRED_METADATA_FIELDS, read_metadata

REQUIRED_STAGE_TIMING_COLUMNS = (
    "setup_elapsed_s",
    "warmup_elapsed_s",
    "measure_elapsed_s",
    "export_elapsed_s",
)
REQUIRED_RESOURCE_COLUMNS = (
    "resource_cpu_vcpu",
    "resource_gpu_count",
    "resource_ram_gib",
    "resource_disk_tb",
    "rhu_rate",
)
REQUIRED_RHU_REFERENCES_KEYS = (
    "c_ref_vcpu",
    "g_ref_gpu",
    "r_ref_gib",
    "d_ref_tb",
)
REQUIRED_RESOURCE_PROFILE_KEYS = (
    "cpu_vcpu",
    "gpu_count",
    "ram_gib",
    "disk_tb",
    "rhu_rate",
)
REQUIRED_RUNNER_LOG_FIELDS = (
    "timestamp_utc",
    "run_id",
    "config_fingerprint",
    "repeat_idx",
    "engine",
    "scenario",
    "dataset_bundle",
    "p99_ms",
    "qps",
    "recall_at_10",
    "sla_violation_rate",
    "errors",
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
        path / "logs" / "runner.log",
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
    _validate_non_negative_numeric_columns(frame, REQUIRED_STAGE_TIMING_COLUMNS)

    missing_resource = [col for col in REQUIRED_RESOURCE_COLUMNS if col not in frame.columns]
    if missing_resource:
        raise ValueError(f"results.parquet missing resource columns: {missing_resource}")
    _validate_non_negative_numeric_columns(frame, REQUIRED_RESOURCE_COLUMNS)

    metadata = read_metadata(path / "run_metadata.json")
    missing_keys = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing_keys:
        raise ValueError(f"run_metadata.json missing keys: {missing_keys}")
    _validate_required_rhu_metadata(metadata)

    expected_run_ids = {str(v) for v in frame.get("run_id", pd.Series(dtype=str)).tolist() if str(v)}
    _validate_runner_log(
        path / "logs" / "runner.log",
        expected_config_fingerprint=str(metadata.get("config_fingerprint") or ""),
        expected_run_ids=expected_run_ids,
    )

    return {
        "rows": int(len(frame)),
        "metadata_keys_ok": True,
        "stage_timing_ok": True,
        "resource_fields_ok": True,
        "resource_metadata_ok": True,
        "logs_ok": True,
    }


def _validate_non_negative_numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"results.parquet column `{col}` contains non-numeric values")
        if (values < 0.0).any():
            raise ValueError(f"results.parquet column `{col}` contains negative values")


def _validate_required_rhu_metadata(metadata: Mapping[str, Any]) -> None:
    refs = metadata.get("rhu_references")
    if not isinstance(refs, dict):
        raise ValueError("run_metadata.json missing required RHU metadata mapping: rhu_references")
    missing_refs = [name for name in REQUIRED_RHU_REFERENCES_KEYS if name not in refs]
    if missing_refs:
        raise ValueError(f"run_metadata.json rhu_references missing keys: {missing_refs}")
    for key in REQUIRED_RHU_REFERENCES_KEYS:
        value = refs.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run_metadata.json rhu_references `{key}` must be numeric") from exc
        if numeric <= 0.0:
            raise ValueError(f"run_metadata.json rhu_references `{key}` must be > 0")

    profile = metadata.get("resource_profile")
    if not isinstance(profile, dict):
        raise ValueError("run_metadata.json missing required RHU metadata mapping: resource_profile")
    missing_profile = [name for name in REQUIRED_RESOURCE_PROFILE_KEYS if name not in profile]
    if missing_profile:
        raise ValueError(f"run_metadata.json resource_profile missing keys: {missing_profile}")
    for key in REQUIRED_RESOURCE_PROFILE_KEYS:
        value = profile.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run_metadata.json resource_profile `{key}` must be numeric") from exc
        if numeric < 0.0:
            raise ValueError(f"run_metadata.json resource_profile `{key}` must be >= 0")


def _validate_runner_log(path: Path, *, expected_config_fingerprint: str, expected_run_ids: set[str]) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("logs/runner.log must contain at least one JSON line")

    seen_run_ids: set[str] = set()
    for idx, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"logs/runner.log line {idx} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"logs/runner.log line {idx} must be a JSON object")
        missing = [name for name in REQUIRED_RUNNER_LOG_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"logs/runner.log line {idx} missing keys: {missing}")
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            raise ValueError(f"logs/runner.log line {idx} has empty run_id")
        seen_run_ids.add(run_id)
        if expected_config_fingerprint:
            observed = str(payload.get("config_fingerprint") or "")
            if observed != expected_config_fingerprint:
                raise ValueError(
                    "logs/runner.log line "
                    f"{idx} has config_fingerprint `{observed}` (expected `{expected_config_fingerprint}`)"
                )

    if expected_run_ids and not expected_run_ids.issubset(seen_run_ids):
        missing_ids = sorted(expected_run_ids - seen_run_ids)
        raise ValueError(f"logs/runner.log missing run_id entries: {missing_ids}")


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
