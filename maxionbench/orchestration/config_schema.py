"""Config loader and typed schema for runner orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from maxionbench.metrics.cost_rhu import RHUReferences, RHUWeights
from maxionbench.datasets.loaders.d4_text import DEFAULT_BEIR_SUBSETS


@dataclass(frozen=True)
class RunConfig:
    engine: str = "mock"
    engine_version: str = "0.1.0"
    adapter_options: dict[str, Any] = field(default_factory=dict)
    scenario: str = "s1_ann_frontier"
    dataset_bundle: str = "D1"
    dataset_hash: str = "synthetic-d1-v1"
    dataset_path: str | None = None
    d2_base_fvecs_path: str | None = None
    d2_query_fvecs_path: str | None = None
    d2_gt_ivecs_path: str | None = None
    d4_use_real_data: bool = False
    d4_beir_root: str | None = None
    d4_beir_subsets: list[str] = field(default_factory=lambda: list(DEFAULT_BEIR_SUBSETS))
    d4_beir_split: str = "test"
    d4_crag_path: str | None = None
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
    s2_selectivities: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5])
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
    rrf_k: int = 60
    s4_dense_candidates: int = 200
    s4_bm25_candidates: int = 200
    s5_candidate_budgets: list[int] = field(default_factory=lambda: [50, 200, 1000])
    s5_reranker_model_id: str = "BAAI/bge-reranker-base"
    s5_reranker_revision_tag: str = "2026-03-04"
    s5_reranker_max_seq_len: int = 512
    s5_reranker_precision: str = "fp16"
    s5_reranker_batch_size: int = 32
    s5_reranker_truncation: str = "right"
    s6_dense_a_candidates: int = 200
    s6_dense_b_candidates: int = 200
    s6_bm25_candidates: int = 200

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
    merged = dict(payload)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    cfg = RunConfig(**merged)
    _validate(cfg)
    return cfg


def _validate(cfg: RunConfig) -> None:
    allowed = {
        "s1_ann_frontier",
        "s2_filtered_ann",
        "s3_churn_smooth",
        "s3b_churn_bursty",
        "calibrate_d3",
        "s4_hybrid",
        "s5_rerank",
        "s6_fusion",
    }
    if cfg.scenario not in allowed:
        raise ValueError(f"Unsupported scenario: {cfg.scenario}")
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
    if not cfg.no_retry:
        raise ValueError("Retries must be disabled during timed measurements.")
    if not cfg.quality_targets:
        raise ValueError("quality_targets must not be empty")
    if not cfg.clients_grid:
        raise ValueError("clients_grid must not be empty")
    if not cfg.search_sweep:
        raise ValueError("search_sweep must not be empty")
    if any(client < 1 for client in cfg.clients_grid):
        raise ValueError("clients_grid values must be >= 1")
    if cfg.lambda_req_s <= 0:
        raise ValueError("lambda_req_s must be positive")
    if cfg.maintenance_interval_s <= 0:
        raise ValueError("maintenance_interval_s must be positive")
    if cfg.s3_max_events < 1:
        raise ValueError("s3_max_events must be >= 1")
    if cfg.rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if cfg.s4_dense_candidates < 1 or cfg.s4_bm25_candidates < 1:
        raise ValueError("s4 candidate budgets must be >= 1")
    if not cfg.s5_candidate_budgets:
        raise ValueError("s5_candidate_budgets must not be empty")
    if any(budget < 1 for budget in cfg.s5_candidate_budgets):
        raise ValueError("s5_candidate_budgets values must be >= 1")
    if cfg.s5_reranker_max_seq_len < 1:
        raise ValueError("s5_reranker_max_seq_len must be >= 1")
    if cfg.s5_reranker_batch_size < 1:
        raise ValueError("s5_reranker_batch_size must be >= 1")
    if cfg.s5_reranker_truncation not in {"left", "right"}:
        raise ValueError("s5_reranker_truncation must be left or right")
    if cfg.s6_dense_a_candidates < 1 or cfg.s6_dense_b_candidates < 1 or cfg.s6_bm25_candidates < 1:
        raise ValueError("s6 candidate budgets must be >= 1")
    if cfg.d4_max_docs < 1 or cfg.d4_max_queries < 1:
        raise ValueError("d4_max_docs and d4_max_queries must be >= 1")
    if cfg.d4_use_real_data and not cfg.d4_beir_root and not cfg.d4_crag_path:
        raise ValueError("d4_use_real_data requires at least d4_beir_root or d4_crag_path")
    if cfg.d4_use_real_data and cfg.d4_beir_root and not cfg.d4_beir_subsets:
        raise ValueError("d4_beir_subsets must not be empty when d4_beir_root is set")
