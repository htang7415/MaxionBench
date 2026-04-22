"""Output schema helpers for result rows and run metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

REQUIRED_METADATA_FIELDS = (
    "run_id",
    "timestamp_utc",
    "engine",
    "engine_version",
    "scenario",
    "dataset_bundle",
    "dataset_hash",
    "seed",
    "clients_read",
    "clients_write",
    "quality_target",
    "rtt_baseline_ms_p50",
    "rtt_baseline_ms_p99",
    "rtt_baseline_request_profile",
    "sla_threshold_ms",
    "rhu_weights",
    "dataset_cache_checksums",
)

PINNED_RTT_BASELINE_REQUEST_PROFILE = "healthcheck_plus_query_topk1_zero_vector"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_HARDWARE_RUNTIME_FIELDS = (
    "hostname",
    "platform",
    "apple_silicon_model",
    "macos_version",
    "docker_version",
    "python_version",
    "cpu_count_logical",
    "slurm_job_id",
    "slurm_array_task_id",
    "container_runtime_hint",
    "total_memory_bytes",
    "gpu_count",
)

RUN_STATUS_FILENAME = "run_status.json"
VALID_RUN_STATUSES = ("success", "failed", "cancelled")


@dataclass(frozen=True)
class ResultRow:
    """Row schema for results.parquet."""

    run_id: str
    timestamp_utc: str
    repeat_idx: int
    engine: str
    engine_version: str
    scenario: str
    dataset_bundle: str
    dataset_hash: str
    seed: int
    clients_read: int
    clients_write: int
    quality_target: float
    search_params_json: str
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    rhu_h: float
    sla_threshold_ms: float
    sla_violation_rate: float
    errors: int
    rtt_baseline_ms_p50: float
    rtt_baseline_ms_p99: float
    budget_level: str | None = None
    embedding_model: str | None = None
    task_cost_est: float = float("nan")
    freshness_hit_at_1s: float = float("nan")
    freshness_hit_at_5s: float = float("nan")
    stale_answer_rate_at_5s: float = float("nan")
    p95_visibility_latency_ms: float = float("nan")
    evidence_coverage_at_10: float = float("nan")
    setup_elapsed_s: float = 0.0
    warmup_target_s: float = 0.0
    warmup_elapsed_s: float = 0.0
    warmup_requests: int = 0
    measure_target_s: float = 0.0
    measure_elapsed_s: float = 0.0
    measure_requests: int = 0
    export_elapsed_s: float = 0.0
    resource_cpu_vcpu: float = 0.0
    resource_gpu_count: float = 0.0
    resource_ram_gib: float = 0.0
    resource_disk_tb: float = 0.0
    rhu_rate: float = 0.0


@dataclass(frozen=True)
class RunMetadata:
    """Run-level metadata required by AGENTS.md Section 9."""

    run_id: str
    timestamp_utc: str
    engine: str
    engine_version: str
    scenario: str
    dataset_bundle: str
    dataset_hash: str
    seed: int
    clients_read: int
    clients_write: int
    quality_target: float
    ground_truth_source: str
    ground_truth_metric: str
    ground_truth_k: int
    ground_truth_engine: str
    rtt_baseline_ms_p50: float
    rtt_baseline_ms_p99: float
    rtt_baseline_request_profile: str
    sla_threshold_ms: float
    rhu_weights: Mapping[str, float]
    config_fingerprint: str
    repeats: int
    no_retry: bool
    profile: str = "legacy"
    budget_level: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    c_llm_in: float = 0.0
    clients_read_grid: list[int] | None = None
    quality_targets: list[float] | None = None
    rhu_references: Mapping[str, float] | None = None
    resource_profile: Mapping[str, float] | None = None
    hardware_runtime: Mapping[str, Any] | None = None
    dataset_cache_checksums: list[Mapping[str, str]] = field(default_factory=list)
    gpu_tracks_omitted: bool = False
    gpu_tracks_omission_reason: str | None = None

    def validate(self) -> None:
        if not self.no_retry:
            raise ValueError("Timed measurements must run with retries disabled.")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.c_llm_in < 0:
            raise ValueError("c_llm_in must be >= 0")
        if self.clients_read < 0 or self.clients_write < 0:
            raise ValueError("client counts must be non-negative")
        if self.budget_level is not None and self.budget_level not in {"b0", "b1", "b2"}:
            raise ValueError("budget_level must be one of b0,b1,b2 when provided")
        missing = [name for name in REQUIRED_METADATA_FIELDS if getattr(self, name, None) is None]
        if missing:
            raise ValueError(f"missing required metadata fields: {missing}")
        if sorted(self.rhu_weights.keys()) != ["w_c", "w_d", "w_g", "w_r"]:
            raise ValueError("rhu_weights must include exactly w_c,w_g,w_r,w_d")
        if self.rtt_baseline_request_profile != PINNED_RTT_BASELINE_REQUEST_PROFILE:
            raise ValueError(
                "rtt_baseline_request_profile must equal "
                f"{PINNED_RTT_BASELINE_REQUEST_PROFILE!r}"
            )
        if not self.ground_truth_source.strip():
            raise ValueError("ground_truth_source must be non-empty")
        if not self.ground_truth_metric.strip():
            raise ValueError("ground_truth_metric must be non-empty")
        if not self.ground_truth_engine.strip():
            raise ValueError("ground_truth_engine must be non-empty")
        if self.ground_truth_k < 0:
            raise ValueError("ground_truth_k must be >= 0")
        if self.rhu_references is not None:
            if sorted(self.rhu_references.keys()) != ["c_ref_vcpu", "d_ref_tb", "g_ref_gpu", "r_ref_gib"]:
                raise ValueError("rhu_references must include c_ref_vcpu,g_ref_gpu,r_ref_gib,d_ref_tb")
        if self.resource_profile is not None:
            if sorted(self.resource_profile.keys()) != ["cpu_vcpu", "disk_tb", "gpu_count", "ram_gib", "rhu_rate"]:
                raise ValueError("resource_profile must include cpu_vcpu,gpu_count,ram_gib,disk_tb,rhu_rate")
        if type(self.hardware_runtime) is not dict:
            raise ValueError("hardware_runtime must be provided as a dict")
        missing_hardware_runtime = [name for name in REQUIRED_HARDWARE_RUNTIME_FIELDS if name not in self.hardware_runtime]
        if missing_hardware_runtime:
            raise ValueError(f"hardware_runtime missing keys: {missing_hardware_runtime}")
        if not isinstance(self.dataset_cache_checksums, list):
            raise ValueError("dataset_cache_checksums must be a list")
        for idx, entry in enumerate(self.dataset_cache_checksums):
            if not isinstance(entry, Mapping):
                raise ValueError(f"dataset_cache_checksums[{idx}] must be a mapping")
            for key in ("path_key", "resolved_path", "source", "expected_sha256", "actual_sha256"):
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset_cache_checksums[{idx}] missing non-empty {key}")
            expected = str(entry.get("expected_sha256")).strip().lower()
            actual = str(entry.get("actual_sha256")).strip().lower()
            if not _SHA256_RE.fullmatch(expected) or not _SHA256_RE.fullmatch(actual):
                raise ValueError(f"dataset_cache_checksums[{idx}] sha256 fields must be lowercase 64-char hex")
            if expected != actual:
                raise ValueError(f"dataset_cache_checksums[{idx}] expected_sha256 must equal actual_sha256")
        if self.gpu_tracks_omission_reason is not None and not str(self.gpu_tracks_omission_reason).strip():
            raise ValueError("gpu_tracks_omission_reason must be non-empty when provided")


@dataclass(frozen=True)
class RunStatus:
    """Terminal status for a run directory."""

    status: str
    timestamp_utc: str
    exit_code: int | None = None
    detail: str | None = None

    def validate(self) -> None:
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(f"run status must be one of {list(VALID_RUN_STATUSES)}, got {self.status!r}")
        if not self.timestamp_utc.strip():
            raise ValueError("run status timestamp_utc must be non-empty")
        if self.detail is not None and not str(self.detail).strip():
            raise ValueError("run status detail must be non-empty when provided")


def utc_now_iso() -> str:
    """Return UTC timestamp in stable ISO-8601 format."""

    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def stable_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash resolved config deterministically for reproducibility tracking."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_results_parquet(path: Path, rows: Iterable[ResultRow]) -> None:
    """Write result rows as Parquet with stable column ordering."""

    rows_list = list(rows)
    if not rows_list:
        raise ValueError("cannot write empty results.parquet")
    frame = pd.DataFrame([asdict(row) for row in rows_list])
    ordered_columns = [field.name for field in ResultRow.__dataclass_fields__.values()]
    frame = frame[ordered_columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    """Write run metadata JSON with deterministic key ordering."""

    metadata.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_run_status(path: Path, status: RunStatus) -> None:
    """Write terminal run status JSON with deterministic key ordering."""

    status.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(status), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    """Write resolved config as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)


def read_metadata(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_run_status(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"run status payload must be a mapping: {path}")
    return payload
