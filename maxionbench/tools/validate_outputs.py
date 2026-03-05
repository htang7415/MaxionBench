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
REQUIRED_GROUND_TRUTH_KEYS = (
    "ground_truth_source",
    "ground_truth_metric",
    "ground_truth_k",
    "ground_truth_engine",
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


def validate_run_directory(path: Path, *, strict_schema: bool = True) -> dict[str, Any]:
    warnings: list[str] = []
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
        _raise_or_warn(
            f"results.parquet missing stage timing columns: {missing_timing}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
    stage_timing_ok = not missing_timing
    if stage_timing_ok:
        stage_timing_ok = _validate_non_negative_numeric_columns(
            frame,
            REQUIRED_STAGE_TIMING_COLUMNS,
            strict_schema=strict_schema,
            warnings=warnings,
        )

    missing_resource = [col for col in REQUIRED_RESOURCE_COLUMNS if col not in frame.columns]
    if missing_resource:
        _raise_or_warn(
            f"results.parquet missing resource columns: {missing_resource}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
    resource_fields_ok = not missing_resource
    if resource_fields_ok:
        resource_fields_ok = _validate_non_negative_numeric_columns(
            frame,
            REQUIRED_RESOURCE_COLUMNS,
            strict_schema=strict_schema,
            warnings=warnings,
        )

    metadata = read_metadata(path / "run_metadata.json")
    missing_keys = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
    if missing_keys:
        raise ValueError(f"run_metadata.json missing keys: {missing_keys}")
    ground_truth_metadata_ok = _validate_required_ground_truth_metadata(
        metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    )
    resource_metadata_ok = _validate_required_rhu_metadata(
        metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    )

    log_path = path / "logs" / "runner.log"
    if not log_path.exists():
        if strict_schema:
            raise FileNotFoundError(f"Missing required artifacts: ['{log_path}']")
        warnings.append(f"Missing required artifacts: ['{log_path}']")
        logs_ok = False
    else:
        expected_run_ids = {str(v) for v in frame.get("run_id", pd.Series(dtype=str)).tolist() if str(v)}
        logs_ok = _validate_runner_log(
            log_path,
            expected_config_fingerprint=str(metadata.get("config_fingerprint") or ""),
            expected_run_ids=expected_run_ids,
            strict_schema=strict_schema,
            warnings=warnings,
        )

    return {
        "rows": int(len(frame)),
        "metadata_keys_ok": True,
        "stage_timing_ok": stage_timing_ok,
        "resource_fields_ok": resource_fields_ok,
        "ground_truth_metadata_ok": ground_truth_metadata_ok,
        "resource_metadata_ok": resource_metadata_ok,
        "logs_ok": logs_ok,
        "warnings": warnings,
    }


def _raise_or_warn(message: str, *, strict_schema: bool, warnings: list[str]) -> None:
    if strict_schema:
        raise ValueError(message)
    warnings.append(message)


def _validate_non_negative_numeric_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    ok = True
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            _raise_or_warn(
                f"results.parquet column `{col}` contains non-numeric values",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
        if (values < 0.0).any():
            _raise_or_warn(
                f"results.parquet column `{col}` contains negative values",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
    return ok


def _validate_required_rhu_metadata(
    metadata: Mapping[str, Any],
    *,
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    ok = True
    refs = metadata.get("rhu_references")
    if not isinstance(refs, dict):
        _raise_or_warn(
            "run_metadata.json missing required RHU metadata mapping: rhu_references",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    missing_refs = [name for name in REQUIRED_RHU_REFERENCES_KEYS if name not in refs]
    if missing_refs:
        _raise_or_warn(
            f"run_metadata.json rhu_references missing keys: {missing_refs}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    for key in REQUIRED_RHU_REFERENCES_KEYS:
        value = refs.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            _raise_or_warn(
                f"run_metadata.json rhu_references `{key}` must be numeric",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        if numeric <= 0.0:
            _raise_or_warn(
                f"run_metadata.json rhu_references `{key}` must be > 0",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False

    profile = metadata.get("resource_profile")
    if not isinstance(profile, dict):
        _raise_or_warn(
            "run_metadata.json missing required RHU metadata mapping: resource_profile",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    missing_profile = [name for name in REQUIRED_RESOURCE_PROFILE_KEYS if name not in profile]
    if missing_profile:
        _raise_or_warn(
            f"run_metadata.json resource_profile missing keys: {missing_profile}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    for key in REQUIRED_RESOURCE_PROFILE_KEYS:
        value = profile.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            _raise_or_warn(
                f"run_metadata.json resource_profile `{key}` must be numeric",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        if numeric < 0.0:
            _raise_or_warn(
                f"run_metadata.json resource_profile `{key}` must be >= 0",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
    return ok


def _validate_required_ground_truth_metadata(
    metadata: Mapping[str, Any],
    *,
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    ok = True
    missing = [name for name in REQUIRED_GROUND_TRUTH_KEYS if name not in metadata]
    if missing:
        _raise_or_warn(
            f"run_metadata.json missing ground truth metadata keys: {missing}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False

    for key in ("ground_truth_source", "ground_truth_metric", "ground_truth_engine"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            _raise_or_warn(
                f"run_metadata.json `{key}` must be a non-empty string",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False

    k_value = metadata.get("ground_truth_k")
    try:
        k_numeric = float(k_value)
    except (TypeError, ValueError):
        _raise_or_warn(
            "run_metadata.json `ground_truth_k` must be numeric",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    if not k_numeric.is_integer():
        _raise_or_warn(
            "run_metadata.json `ground_truth_k` must be an integer value",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    if k_numeric < 0:
        _raise_or_warn(
            "run_metadata.json `ground_truth_k` must be >= 0",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    return ok


def _validate_runner_log(
    path: Path,
    *,
    expected_config_fingerprint: str,
    expected_run_ids: set[str],
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        _raise_or_warn(
            "logs/runner.log must contain at least one JSON line",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False

    ok = True
    seen_run_ids: set[str] = set()
    for idx, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _raise_or_warn(
                f"logs/runner.log line {idx} is not valid JSON",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        if not isinstance(payload, dict):
            _raise_or_warn(
                f"logs/runner.log line {idx} must be a JSON object",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        missing = [name for name in REQUIRED_RUNNER_LOG_FIELDS if name not in payload]
        if missing:
            _raise_or_warn(
                f"logs/runner.log line {idx} missing keys: {missing}",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            _raise_or_warn(
                f"logs/runner.log line {idx} has empty run_id",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        seen_run_ids.add(run_id)
        if expected_config_fingerprint:
            observed = str(payload.get("config_fingerprint") or "")
            if observed != expected_config_fingerprint:
                _raise_or_warn(
                    "logs/runner.log line "
                    f"{idx} has config_fingerprint `{observed}` (expected `{expected_config_fingerprint}`)",
                    strict_schema=strict_schema,
                    warnings=warnings,
                )
                ok = False

    if expected_run_ids and not expected_run_ids.issubset(seen_run_ids):
        missing_ids = sorted(expected_run_ids - seen_run_ids)
        _raise_or_warn(
            f"logs/runner.log missing run_id entries: {missing_ids}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    return ok


def validate_path(path: Path, *, strict_schema: bool = True) -> dict[str, Any]:
    run_dirs = discover_run_directories(path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {path.resolve()}")
    runs: list[dict[str, Any]] = []
    total_rows = 0
    warnings: list[str] = []
    for run_dir in run_dirs:
        summary = validate_run_directory(run_dir, strict_schema=strict_schema)
        summary["run_path"] = str(run_dir)
        total_rows += int(summary["rows"])
        warnings.extend([str(item) for item in summary.get("warnings", [])])
        runs.append(summary)
    return {
        "input": str(path.resolve()),
        "strict_schema": strict_schema,
        "warning_count": len(warnings),
        "warnings": warnings,
        "run_count": len(runs),
        "total_rows": total_rows,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate MaxionBench outputs")
    parser.add_argument("--input", required=True, help="Run directory to validate")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--strict-schema",
        action="store_true",
        help="Enforce strict schema validation (default).",
    )
    mode_group.add_argument(
        "--legacy-ok",
        action="store_true",
        help="Allow legacy schema drift (stage/resource/log checks become warnings).",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args(argv)
    summary = validate_path(Path(args.input).resolve(), strict_schema=not bool(args.legacy_ok))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
