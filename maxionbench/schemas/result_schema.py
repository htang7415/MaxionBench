"""Output schema helpers for result rows and run metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
    "sla_threshold_ms",
    "rhu_weights",
)

REQUIRED_HARDWARE_RUNTIME_FIELDS = (
    "hostname",
    "platform",
    "python_version",
    "cpu_count_logical",
    "slurm_job_id",
    "slurm_array_task_id",
    "container_runtime_hint",
    "total_memory_bytes",
    "gpu_count",
)


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
    sla_threshold_ms: float
    rhu_weights: Mapping[str, float]
    config_fingerprint: str
    repeats: int
    no_retry: bool
    clients_read_grid: list[int] | None = None
    quality_targets: list[float] | None = None
    rhu_references: Mapping[str, float] | None = None
    resource_profile: Mapping[str, float] | None = None
    hardware_runtime: Mapping[str, Any] | None = None
    gpu_tracks_omitted: bool = False
    gpu_tracks_omission_reason: str | None = None

    def validate(self) -> None:
        if not self.no_retry:
            raise ValueError("Timed measurements must run with retries disabled.")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.clients_read < 0 or self.clients_write < 0:
            raise ValueError("client counts must be non-negative")
        missing = [name for name in REQUIRED_METADATA_FIELDS if getattr(self, name, None) is None]
        if missing:
            raise ValueError(f"missing required metadata fields: {missing}")
        if sorted(self.rhu_weights.keys()) != ["w_c", "w_d", "w_g", "w_r"]:
            raise ValueError("rhu_weights must include exactly w_c,w_g,w_r,w_d")
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
        if not isinstance(self.hardware_runtime, Mapping):
            raise ValueError("hardware_runtime must be provided as a mapping")
        missing_hardware_runtime = [name for name in REQUIRED_HARDWARE_RUNTIME_FIELDS if name not in self.hardware_runtime]
        if missing_hardware_runtime:
            raise ValueError(f"hardware_runtime missing keys: {missing_hardware_runtime}")
        if self.gpu_tracks_omission_reason is not None and not str(self.gpu_tracks_omission_reason).strip():
            raise ValueError("gpu_tracks_omission_reason must be non-empty when provided")


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


def write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    """Write resolved config as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)


def read_metadata(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
