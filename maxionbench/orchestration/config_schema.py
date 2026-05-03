"""Config loader and typed schema for runner orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from maxionbench.metrics.cost_rhu import RHUReferences, RHUWeights
from maxionbench.datasets.loaders.d4_text import DEFAULT_BEIR_SUBSETS

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_TOKEN_RE = re.compile(
    r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<braced_default>[^}]*))?\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$"
)
_ENV_SUB_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<braced_default>[^}]*))?\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    profile: str = "maxionbench"
    budget_level: str | None = None
    engine: str = "mock"
    engine_version: str = "0.1.0"
    embedding_model: str | None = None
    embedding_dim: int | None = None
    c_llm_in: float = 0.0
    adapter_options: dict[str, Any] = field(default_factory=dict)
    index_params: dict[str, Any] = field(default_factory=dict)
    scenario: str = "s1_single_hop"
    dataset_bundle: str = "D4"
    dataset_hash: str = "portable-d4-v1"
    dataset_path: str | None = None
    processed_dataset_path: str | None = None
    processed_dataset_sha256: str | None = None
    dataset_path_sha256: str | None = None
    d2_base_fvecs_path: str | None = None
    d2_base_fvecs_sha256: str | None = None
    d2_query_fvecs_path: str | None = None
    d2_query_fvecs_sha256: str | None = None
    d2_gt_ivecs_path: str | None = None
    d2_gt_ivecs_sha256: str | None = None
    d4_use_real_data: bool = False
    d4_beir_root: str | None = None
    d4_beir_subsets: list[str] = field(default_factory=lambda: list(DEFAULT_BEIR_SUBSETS))
    d4_beir_split: str = "test"
    d4_crag_source: str = "facebookresearch/CRAG"
    d4_crag_path: str | None = None
    d4_crag_sha256: str | None = None
    d4_include_crag: bool = True
    d4_max_docs: int = 200000
    d4_max_queries: int = 5000
    seed: int = 42
    repeats: int = 3
    no_retry: bool = True
    output_dir: str = "artifacts/runs/default"
    quality_target: float = 0.8
    quality_targets: list[float] = field(default_factory=lambda: [0.80, 0.90, 0.95])
    clients_read: int = 1
    clients_write: int = 0
    clients_grid: list[int] = field(default_factory=lambda: [1, 8, 32, 64])
    search_sweep: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"hnsw_ef": 64},
            {"hnsw_ef": 128},
            {"hnsw_ef": 256},
        ]
    )
    phase_timing_mode: str = "bounded"
    phase_max_requests_per_phase: int | None = None
    warmup_s: int = 120
    steady_state_s: int = 300
    rpc_baseline_requests: int = 1000
    sla_threshold_ms: float = 50.0
    vector_dim: int = 64
    num_vectors: int = 5000
    num_queries: int = 200
    top_k: int = 10
    c_ref_vcpu: float = 96.0
    g_ref_gpu: float = 1.0
    r_ref_gib: float = 512.0
    d_ref_tb: float = 7.68
    w_c: float = 0.25
    w_g: float = 0.25
    w_r: float = 0.25
    w_d: float = 0.25
    output_d3_params_path: str = "artifacts/calibration/d3_params.yaml"
    d3_k_clusters: int = 4096
    d3_num_tenants: int = 100
    d3_num_acl_buckets: int = 16
    d3_num_time_buckets: int = 52
    d3_beta_tenant: float = 0.75
    d3_beta_acl: float = 0.70
    d3_beta_time: float = 0.65
    d3_seed: int = 42
    calibration_require_real_data: bool = False
    s2_selectivities: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5])
    s2_max_freshness_events: int | None = None
    lambda_req_s: float = 1000.0
    s3_read_rate: float = 800.0
    s3_insert_rate: float = 100.0
    s3_update_rate: float = 50.0
    s3_delete_rate: float = 50.0
    maintenance_interval_s: float = 60.0
    s3_max_events: int = 5000
    s3b_on_s: float = 30.0
    s3b_off_s: float = 90.0
    s3b_on_write_mult: float = 8.0
    s3b_off_write_mult: float = 0.25
    allow_missing_s3_baseline: bool = False
    allow_unverified_d3_params: bool = False
    d3_min_calibration_vectors: int = 10_000_000
    rrf_k: int = 60
    s4_dense_candidates: int = 200
    s4_bm25_candidates: int = 200

    @property
    def references(self) -> RHUReferences:
        return RHUReferences(
            c_ref_vcpu=self.c_ref_vcpu,
            g_ref_gpu=self.g_ref_gpu,
            r_ref_gib=self.r_ref_gib,
            d_ref_tb=self.d_ref_tb,
        )

    @property
    def weights(self) -> RHUWeights:
        return RHUWeights(w_c=self.w_c, w_g=self.w_g, w_r=self.w_r, w_d=self.w_d)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_run_config(path: Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    merged = expand_env_placeholders(payload)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    merged = _normalize_engine_specific_payload(merged)
    cfg = RunConfig(**merged)
    _validate(cfg)
    return cfg


def _normalize_engine_specific_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    index_params = normalized.get("index_params")
    if isinstance(index_params, Mapping):
        normalized["index_params"] = dict(index_params)
    engine = str(normalized.get("engine", "")).strip().lower()
    if engine != "pgvector":
        return normalized

    search_sweep = normalized.get("search_sweep")
    if not isinstance(search_sweep, list):
        return normalized

    adapter_options = normalized.get("adapter_options")
    index_method = "ivfflat"
    if isinstance(adapter_options, Mapping):
        raw_method = adapter_options.get("index_method")
        if raw_method is not None and str(raw_method).strip():
            index_method = str(raw_method).strip().lower()

    target_key = "hnsw_ef_search" if index_method == "hnsw" else "ivfflat_probes"
    normalized_sweep: list[dict[str, Any]] = []
    changed = False
    for item in search_sweep:
        if not isinstance(item, dict):
            normalized_sweep.append(item)
            continue
        if (
            "hnsw_ef" in item
            and "ivfflat_probes" not in item
            and "hnsw_ef_search" not in item
        ):
            converted = dict(item)
            converted[target_key] = converted.pop("hnsw_ef")
            normalized_sweep.append(converted)
            changed = True
            continue
        normalized_sweep.append(item)

    if changed:
        normalized["search_sweep"] = normalized_sweep
    return normalized


def _validate(cfg: RunConfig) -> None:
    allowed = {
        "s1_single_hop",
        "s2_streaming_memory",
        "s3_multi_hop",
    }
    if cfg.scenario not in allowed:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")
    if cfg.budget_level is not None and cfg.budget_level not in {"b0", "b1", "b2"}:
        raise ValueError("budget_level must be one of b0,b1,b2 when provided")
    if cfg.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if cfg.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if cfg.vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    if cfg.num_vectors < 1:
        raise ValueError("num_vectors must be >= 1")
    if cfg.num_queries < 1:
        raise ValueError("num_queries must be >= 1")
    if cfg.warmup_s < 0:
        raise ValueError("warmup_s must be >= 0")
    if cfg.steady_state_s <= 0:
        raise ValueError("steady_state_s must be > 0")
    if cfg.phase_timing_mode not in {"bounded", "strict"}:
        raise ValueError("phase_timing_mode must be bounded or strict")
    if cfg.phase_max_requests_per_phase is not None and cfg.phase_max_requests_per_phase < 1:
        raise ValueError("phase_max_requests_per_phase must be >= 1 when set")
    if cfg.s2_max_freshness_events is not None and cfg.s2_max_freshness_events < 1:
        raise ValueError("s2_max_freshness_events must be >= 1 when set")
    if not cfg.no_retry:
        raise ValueError("Retries must be disabled during timed measurements.")
    if not cfg.quality_targets:
        raise ValueError("quality_targets must not be empty")
    if not cfg.clients_grid:
        raise ValueError("clients_grid must not be empty")
    if not cfg.search_sweep:
        raise ValueError("search_sweep must not be empty")
    if not isinstance(cfg.index_params, dict):
        raise ValueError("index_params must be a mapping")
    if any(client < 1 for client in cfg.clients_grid):
        raise ValueError("clients_grid values must be >= 1")
    if cfg.lambda_req_s <= 0:
        raise ValueError("lambda_req_s must be positive")
    if cfg.maintenance_interval_s <= 0:
        raise ValueError("maintenance_interval_s must be positive")
    if cfg.s3_max_events < 1:
        raise ValueError("s3_max_events must be >= 1")
    if cfg.d3_min_calibration_vectors < 1:
        raise ValueError("d3_min_calibration_vectors must be >= 1")
    if not isinstance(cfg.calibration_require_real_data, bool):
        raise ValueError("calibration_require_real_data must be a boolean")
    if cfg.rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if cfg.s4_dense_candidates < 1 or cfg.s4_bm25_candidates < 1:
        raise ValueError("s4 candidate budgets must be >= 1")
    if cfg.d4_max_docs < 1 or cfg.d4_max_queries < 1:
        raise ValueError("d4_max_docs and d4_max_queries must be >= 1")
    if cfg.c_llm_in < 0:
        raise ValueError("c_llm_in must be >= 0")
    if cfg.embedding_dim is not None and cfg.embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1 when provided")
    if cfg.d4_use_real_data and not cfg.processed_dataset_path and not cfg.d4_beir_root and not cfg.d4_crag_path:
        raise ValueError("d4_use_real_data requires at least one of processed_dataset_path, d4_beir_root, or d4_crag_path")
    if cfg.d4_use_real_data and cfg.d4_beir_root and not cfg.d4_beir_subsets:
        raise ValueError("d4_beir_subsets must not be empty when d4_beir_root is set")
    for key in [
        "processed_dataset_sha256",
        "dataset_path_sha256",
        "d2_base_fvecs_sha256",
        "d2_query_fvecs_sha256",
        "d2_gt_ivecs_sha256",
        "d4_crag_sha256",
    ]:
        value = getattr(cfg, key)
        if value is None:
            continue
        text = str(value).strip().lower()
        if not _SHA256_RE.fullmatch(text):
            raise ValueError(f"{key} must be a 64-character lowercase hex sha256 string when provided")


def expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    token = value.strip()
    match = _ENV_TOKEN_RE.fullmatch(token)
    if match is not None:
        env_name = match.group("braced") or match.group("plain") or ""
        default_value = match.group("braced_default") if match.group("braced") else None
        env_value = _resolve_env_value(env_name, default_value)
        if env_value in {None, ""}:
            _LOG.warning("Environment placeholder %s resolved to empty/None; using null", env_name)
            return None
        return env_value

    def _replace(match_obj: re.Match[str]) -> str:
        env_name = match_obj.group("braced") or match_obj.group("plain") or ""
        default_value = match_obj.group("braced_default") if match_obj.group("braced") else None
        env_value = _resolve_env_value(env_name, default_value)
        if env_value is None:
            return match_obj.group(0)
        return env_value

    return _ENV_SUB_RE.sub(_replace, value)


def _resolve_env_value(env_name: str, default_value: str | None) -> str | None:
    env_value = os.environ.get(env_name)
    if env_value not in {None, ""}:
        return env_value
    if default_value is not None:
        return default_value
    return None
