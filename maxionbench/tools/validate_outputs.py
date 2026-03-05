"""Validate run artifacts for schema compliance."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from maxionbench.schemas.result_schema import REQUIRED_HARDWARE_RUNTIME_FIELDS, REQUIRED_METADATA_FIELDS, read_metadata

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
HARDWARE_RUNTIME_NUMERIC_NON_NEGATIVE_KEYS = (
    "cpu_count_logical",
    "total_memory_bytes",
    "gpu_count",
)
HARDWARE_RUNTIME_REQUIRED_STRING_KEYS = (
    "hostname",
    "platform",
    "python_version",
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
PINNED_WARMUP_S = 120.0
PINNED_STEADY_STATE_S = 300.0
PINNED_RPC_BASELINE_REQUESTS = 1000
PINNED_REPEATS_MIN = 3
PINNED_REPEATS_MAX = 5
PINNED_PROTOCOL_SKIP_SCENARIOS = {"calibrate_d3"}
PINNED_S1_CLIENTS_GRID = [1, 8, 32, 64]
PINNED_S1_QUALITY_TARGETS = [0.8, 0.9, 0.95]
PINNED_S2_SELECTIVITIES = [0.001, 0.01, 0.1, 0.5]
PINNED_S5_CANDIDATE_BUDGETS = [50, 200, 1000]
PINNED_S5_MODEL_ID = "BAAI/bge-reranker-base"
PINNED_S5_REVISION_TAG = "2026-03-04"
PINNED_S5_MAX_SEQ_LEN = 512
PINNED_S5_PRECISION = "fp16"
PINNED_S5_BATCH_SIZE = 32
PINNED_S5_TRUNCATION = "right"


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


def validate_run_directory(
    path: Path,
    *,
    strict_schema: bool = True,
    enforce_protocol: bool = False,
) -> dict[str, Any]:
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
    config_payload = _read_resolved_config(path / "config_resolved.yaml")
    ground_truth_metadata_ok = _validate_required_ground_truth_metadata(
        metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    )
    hardware_runtime_ok = _validate_required_hardware_runtime(
        metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    )
    resource_metadata_ok = _validate_required_rhu_metadata(
        metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    )
    runtime_protocol_ok = _validate_runtime_protocol(
        frame=frame,
        metadata=metadata,
        config_payload=config_payload,
        strict_schema=strict_schema,
        warnings=warnings,
        enforce_protocol=enforce_protocol,
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
        "hardware_runtime_ok": hardware_runtime_ok,
        "resource_metadata_ok": resource_metadata_ok,
        "runtime_protocol_ok": runtime_protocol_ok,
        "logs_ok": logs_ok,
        "warnings": warnings,
    }


def _read_resolved_config(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"config_resolved.yaml is not valid YAML: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("config_resolved.yaml root must be a mapping")
    return payload


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


def _validate_required_hardware_runtime(
    metadata: Mapping[str, Any],
    *,
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    payload = metadata.get("hardware_runtime")
    if not isinstance(payload, Mapping):
        _raise_or_warn(
            "run_metadata.json missing required hardware/runtime summary mapping: hardware_runtime",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False

    ok = True
    missing = [name for name in REQUIRED_HARDWARE_RUNTIME_FIELDS if name not in payload]
    if missing:
        _raise_or_warn(
            f"run_metadata.json hardware_runtime missing keys: {missing}",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False

    for key in HARDWARE_RUNTIME_REQUIRED_STRING_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            _raise_or_warn(
                f"run_metadata.json hardware_runtime `{key}` must be a non-empty string",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False

    for key in HARDWARE_RUNTIME_NUMERIC_NON_NEGATIVE_KEYS:
        value = payload.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            _raise_or_warn(
                f"run_metadata.json hardware_runtime `{key}` must be numeric",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            ok = False
            continue
        if numeric < 0.0:
            _raise_or_warn(
                f"run_metadata.json hardware_runtime `{key}` must be >= 0",
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


def _validate_runtime_protocol(
    *,
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    strict_schema: bool,
    warnings: list[str],
    enforce_protocol: bool,
) -> bool:
    if not enforce_protocol:
        return True

    scenario = str(metadata.get("scenario") or "")
    if scenario in PINNED_PROTOCOL_SKIP_SCENARIOS:
        return True

    ok = True
    if metadata.get("no_retry") is not True:
        _raise_or_warn(
            "run_metadata.json `no_retry` must be true for pinned protocol runs",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False

    repeats_value = metadata.get("repeats")
    try:
        repeats = int(repeats_value)
    except (TypeError, ValueError):
        _raise_or_warn(
            "run_metadata.json `repeats` must be an integer for pinned protocol runs",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        repeats = PINNED_REPEATS_MIN - 1
        ok = False
    if not (PINNED_REPEATS_MIN <= repeats <= PINNED_REPEATS_MAX):
        _raise_or_warn(
            f"run_metadata.json `repeats` must be in [{PINNED_REPEATS_MIN}, {PINNED_REPEATS_MAX}] for pinned protocol runs",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False

    ok = _validate_frame_constant(
        frame=frame,
        column="warmup_target_s",
        expected=PINNED_WARMUP_S,
        strict_schema=strict_schema,
        warnings=warnings,
        label="results.parquet warmup target",
    ) and ok
    ok = _validate_frame_constant(
        frame=frame,
        column="measure_target_s",
        expected=PINNED_STEADY_STATE_S,
        strict_schema=strict_schema,
        warnings=warnings,
        label="results.parquet steady-state target",
    ) and ok

    ok = _validate_config_constant(
        config_payload=config_payload,
        key="warmup_s",
        expected=PINNED_WARMUP_S,
        strict_schema=strict_schema,
        warnings=warnings,
        label="config_resolved warmup_s",
    ) and ok
    ok = _validate_config_constant(
        config_payload=config_payload,
        key="steady_state_s",
        expected=PINNED_STEADY_STATE_S,
        strict_schema=strict_schema,
        warnings=warnings,
        label="config_resolved steady_state_s",
    ) and ok
    ok = _validate_config_constant(
        config_payload=config_payload,
        key="rpc_baseline_requests",
        expected=PINNED_RPC_BASELINE_REQUESTS,
        strict_schema=strict_schema,
        warnings=warnings,
        label="config_resolved rpc_baseline_requests",
    ) and ok
    ok = _validate_scenario_protocol_pins(
        scenario=scenario,
        config_payload=config_payload,
        metadata=metadata,
        strict_schema=strict_schema,
        warnings=warnings,
    ) and ok
    ok = _validate_gpu_track_omission_metadata(
        metadata=metadata,
        config_payload=config_payload,
        strict_schema=strict_schema,
        warnings=warnings,
    ) and ok
    return ok


def _validate_gpu_track_omission_metadata(
    *,
    metadata: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    if "gpu_tracks_omitted" not in metadata:
        _raise_or_warn(
            "run_metadata.json missing `gpu_tracks_omitted` boolean for pinned protocol runs",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    omitted_value = metadata.get("gpu_tracks_omitted")
    if not isinstance(omitted_value, bool):
        _raise_or_warn(
            "run_metadata.json `gpu_tracks_omitted` must be a boolean",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False

    reason_value = metadata.get("gpu_tracks_omission_reason")
    if omitted_value:
        if not isinstance(reason_value, str) or not reason_value.strip():
            _raise_or_warn(
                "run_metadata.json must provide non-empty `gpu_tracks_omission_reason` when gpu_tracks_omitted=true",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            return False
    elif reason_value not in {None, ""} and not (isinstance(reason_value, str) and not reason_value.strip()):
        _raise_or_warn(
            "run_metadata.json `gpu_tracks_omission_reason` must be empty/null when gpu_tracks_omitted=false",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False

    readiness = config_payload.get("readiness")
    allow_gpu_unavailable = False
    if isinstance(readiness, Mapping):
        allow_gpu_unavailable = bool(readiness.get("allow_gpu_unavailable", False))
    hardware = metadata.get("hardware_runtime")
    observed_gpu_count = 0.0
    if isinstance(hardware, Mapping):
        try:
            observed_gpu_count = float(hardware.get("gpu_count", 0.0))
        except (TypeError, ValueError):
            observed_gpu_count = 0.0
    expected_omitted = allow_gpu_unavailable and observed_gpu_count <= 0.0
    if omitted_value != expected_omitted:
        _raise_or_warn(
            "run_metadata.json `gpu_tracks_omitted` does not match readiness.allow_gpu_unavailable "
            f"({allow_gpu_unavailable}) and observed hardware_runtime.gpu_count ({observed_gpu_count})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    return True


def _validate_frame_constant(
    *,
    frame: pd.DataFrame,
    column: str,
    expected: float,
    strict_schema: bool,
    warnings: list[str],
    label: str,
) -> bool:
    if column not in frame.columns:
        _raise_or_warn(
            f"{label} missing column `{column}`",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        _raise_or_warn(
            f"{label} column `{column}` must be numeric",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    observed = sorted({float(v) for v in values.tolist()})
    if observed != [float(expected)]:
        _raise_or_warn(
            f"{label} column `{column}` must be exactly {expected} (observed {observed})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    return True


def _validate_config_constant(
    *,
    config_payload: Mapping[str, Any],
    key: str,
    expected: Any,
    strict_schema: bool,
    warnings: list[str],
    label: str,
) -> bool:
    value = config_payload.get(key)
    if isinstance(expected, str):
        observed = str(value) if value is not None else ""
        if observed != expected:
            _raise_or_warn(
                f"{label} `{key}` must be exactly {expected} (observed {observed})",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            return False
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _raise_or_warn(
            f"{label} `{key}` must be numeric",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    if numeric != float(expected):
        _raise_or_warn(
            f"{label} `{key}` must be exactly {expected} (observed {numeric})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    return True


def _validate_scenario_protocol_pins(
    *,
    scenario: str,
    config_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    checks: list[tuple[str, Any]] = []
    if scenario == "s1_ann_frontier":
        checks = [
            ("clients_grid", PINNED_S1_CLIENTS_GRID),
            ("quality_targets", PINNED_S1_QUALITY_TARGETS),
            ("sla_threshold_ms", 50.0),
        ]
    elif scenario == "s2_filtered_ann":
        checks = [
            ("clients_read", 32),
            ("clients_grid", [32]),
            ("s2_selectivities", PINNED_S2_SELECTIVITIES),
            ("sla_threshold_ms", 80.0),
        ]
    elif scenario == "s3_churn_smooth":
        checks = [
            ("clients_read", 32),
            ("clients_write", 8),
            ("lambda_req_s", 1000.0),
            ("s3_read_rate", 800.0),
            ("s3_insert_rate", 100.0),
            ("s3_update_rate", 50.0),
            ("s3_delete_rate", 50.0),
            ("maintenance_interval_s", 60.0),
            ("sla_threshold_ms", 120.0),
        ]
    elif scenario == "s3b_churn_bursty":
        checks = [
            ("clients_read", 32),
            ("clients_write", 8),
            ("lambda_req_s", 1000.0),
            ("s3_read_rate", 800.0),
            ("s3_insert_rate", 100.0),
            ("s3_update_rate", 50.0),
            ("s3_delete_rate", 50.0),
            ("maintenance_interval_s", 60.0),
            ("s3b_on_s", 30.0),
            ("s3b_off_s", 90.0),
            ("s3b_on_write_mult", 8.0),
            ("s3b_off_write_mult", 0.25),
            ("sla_threshold_ms", 120.0),
        ]
    elif scenario == "s4_hybrid":
        checks = [
            ("clients_read", 16),
            ("clients_write", 0),
            ("rrf_k", 60),
            ("s4_dense_candidates", 200),
            ("s4_bm25_candidates", 200),
            ("sla_threshold_ms", 150.0),
        ]
    elif scenario == "s5_rerank":
        checks = [
            ("clients_read", 16),
            ("clients_write", 0),
            ("s5_candidate_budgets", PINNED_S5_CANDIDATE_BUDGETS),
            ("s5_reranker_model_id", PINNED_S5_MODEL_ID),
            ("s5_reranker_revision_tag", PINNED_S5_REVISION_TAG),
            ("s5_reranker_max_seq_len", PINNED_S5_MAX_SEQ_LEN),
            ("s5_reranker_precision", PINNED_S5_PRECISION),
            ("s5_reranker_batch_size", PINNED_S5_BATCH_SIZE),
            ("s5_reranker_truncation", PINNED_S5_TRUNCATION),
            ("sla_threshold_ms", 300.0),
        ]
    elif scenario == "s6_fusion":
        checks = [
            ("clients_read", 16),
            ("clients_write", 0),
            ("rrf_k", 60),
            ("s6_dense_a_candidates", 200),
            ("s6_dense_b_candidates", 200),
            ("s6_bm25_candidates", 200),
            ("sla_threshold_ms", 180.0),
        ]
    else:
        return True

    ok = True
    for key, expected in checks:
        if isinstance(expected, list):
            if not _validate_config_list_constant(
                config_payload=config_payload,
                key=key,
                expected=expected,
                strict_schema=strict_schema,
                warnings=warnings,
                label="config_resolved scenario pin",
            ):
                ok = False
            continue
        if not _validate_config_constant(
            config_payload=config_payload,
            key=key,
            expected=float(expected) if isinstance(expected, (int, float)) else expected,
            strict_schema=strict_schema,
            warnings=warnings,
            label="config_resolved scenario pin",
        ):
            ok = False
    if scenario == "s1_ann_frontier":
        if not _validate_metadata_list_constant(
            metadata=metadata,
            key="clients_read_grid",
            expected=PINNED_S1_CLIENTS_GRID,
            strict_schema=strict_schema,
            warnings=warnings,
            label="run_metadata scenario pin",
        ):
            ok = False
        if not _validate_metadata_list_constant(
            metadata=metadata,
            key="quality_targets",
            expected=PINNED_S1_QUALITY_TARGETS,
            strict_schema=strict_schema,
            warnings=warnings,
            label="run_metadata scenario pin",
        ):
            ok = False
    return ok


def _validate_config_list_constant(
    *,
    config_payload: Mapping[str, Any],
    key: str,
    expected: list[Any],
    strict_schema: bool,
    warnings: list[str],
    label: str,
) -> bool:
    value = config_payload.get(key)
    if not isinstance(value, list):
        _raise_or_warn(
            f"{label} `{key}` must be a list",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    observed = [_normalize_constant_value(item) for item in value]
    normalized_expected = [_normalize_constant_value(item) for item in expected]
    if observed != normalized_expected:
        _raise_or_warn(
            f"{label} `{key}` must be exactly {normalized_expected} (observed {observed})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    return True


def _normalize_constant_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _validate_metadata_list_constant(
    *,
    metadata: Mapping[str, Any],
    key: str,
    expected: list[Any],
    strict_schema: bool,
    warnings: list[str],
    label: str,
) -> bool:
    value = metadata.get(key)
    if not isinstance(value, list):
        _raise_or_warn(
            f"{label} `{key}` must be a list",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    observed = [_normalize_constant_value(item) for item in value]
    normalized_expected = [_normalize_constant_value(item) for item in expected]
    if observed != normalized_expected:
        _raise_or_warn(
            f"{label} `{key}` must be exactly {normalized_expected} (observed {observed})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        return False
    return True


def validate_path(
    path: Path,
    *,
    strict_schema: bool = True,
    enforce_protocol: bool = False,
) -> dict[str, Any]:
    run_dirs = discover_run_directories(path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {path.resolve()}")
    runs: list[dict[str, Any]] = []
    total_rows = 0
    warnings: list[str] = []
    for run_dir in run_dirs:
        summary = validate_run_directory(
            run_dir,
            strict_schema=strict_schema,
            enforce_protocol=enforce_protocol,
        )
        summary["run_path"] = str(run_dir)
        total_rows += int(summary["rows"])
        warnings.extend([str(item) for item in summary.get("warnings", [])])
        runs.append(summary)
    cross_run_protocol_ok = True
    if enforce_protocol:
        cross_run_protocol_ok = _validate_cross_run_protocol(
            run_dirs=run_dirs,
            strict_schema=strict_schema,
            warnings=warnings,
        )
    return {
        "input": str(path.resolve()),
        "strict_schema": strict_schema,
        "enforce_protocol": enforce_protocol,
        "cross_run_protocol_ok": cross_run_protocol_ok,
        "warning_count": len(warnings),
        "warnings": warnings,
        "run_count": len(runs),
        "total_rows": total_rows,
        "runs": runs,
    }


def _validate_cross_run_protocol(
    *,
    run_dirs: list[Path],
    strict_schema: bool,
    warnings: list[str],
) -> bool:
    records: list[tuple[Path, Mapping[str, Any]]] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        try:
            metadata = read_metadata(metadata_path)
        except Exception as exc:
            _raise_or_warn(
                f"failed to read {metadata_path}: {exc}",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            return False
        records.append((run_dir, metadata))

    s1_baselines: set[tuple[str, str, int]] = set()
    s3_like_runs: list[tuple[Path, str, tuple[str, str, int]]] = []
    for run_dir, metadata in records:
        scenario = str(metadata.get("scenario") or "")
        engine = str(metadata.get("engine") or "")
        dataset_bundle = str(metadata.get("dataset_bundle") or "")
        try:
            clients_read = int(metadata.get("clients_read"))
        except (TypeError, ValueError):
            _raise_or_warn(
                f"run_metadata.json for `{run_dir}` must include integer clients_read",
                strict_schema=strict_schema,
                warnings=warnings,
            )
            return False
        key = (engine, dataset_bundle, clients_read)
        if scenario == "s1_ann_frontier":
            s1_baselines.add(key)
        elif scenario in {"s3_churn_smooth", "s3b_churn_bursty"}:
            s3_like_runs.append((run_dir, scenario, key))

    ok = True
    for run_dir, scenario, key in s3_like_runs:
        if key in s1_baselines:
            continue
        engine, dataset_bundle, clients_read = key
        _raise_or_warn(
            f"{scenario} run `{run_dir}` is missing matched S1 baseline "
            f"(engine={engine}, dataset_bundle={dataset_bundle}, clients_read={clients_read})",
            strict_schema=strict_schema,
            warnings=warnings,
        )
        ok = False
    return ok


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
    parser.add_argument(
        "--enforce-protocol",
        action="store_true",
        help="Enforce pinned runtime protocol values (warmup/steady/repeats/RPC baseline).",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args(argv)
    summary = validate_path(
        Path(args.input).resolve(),
        strict_schema=not bool(args.legacy_ok),
        enforce_protocol=bool(args.enforce_protocol),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
