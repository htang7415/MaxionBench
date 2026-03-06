from __future__ import annotations

import json

import numpy as np

from maxionbench.adapters.mock import MockAdapter
from maxionbench.scenarios import s5_rerank as s5_mod


class _CountingMockAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_upsert_calls = 0

    def bulk_upsert(self, records):  # type: ignore[override]
        self.bulk_upsert_calls += 1
        return super().bulk_upsert(records)


def _base_cfg() -> s5_mod.S5Config:
    return s5_mod.S5Config(
        vector_dim=16,
        num_vectors=160,
        num_queries=24,
        top_k=10,
        clients_read=4,
        sla_threshold_ms=300.0,
        candidate_budgets=[20],
        warmup_s=0.01,
        steady_state_s=0.01,
        phase_timing_mode="strict",
        phase_max_requests_per_phase=12,
    )


def _run_with_runtime(
    runtime: s5_mod._RerankerRuntime,
    *,
    seed: int = 7,
    require_hf_backend: bool = False,
) -> s5_mod.S5BudgetResult:
    adapter = MockAdapter()
    adapter.create(collection="s5-test", dimension=16, metric="ip")
    cfg_base = _base_cfg()
    cfg = s5_mod.S5Config(
        vector_dim=cfg_base.vector_dim,
        num_vectors=cfg_base.num_vectors,
        num_queries=cfg_base.num_queries,
        top_k=cfg_base.top_k,
        clients_read=cfg_base.clients_read,
        sla_threshold_ms=cfg_base.sla_threshold_ms,
        candidate_budgets=list(cfg_base.candidate_budgets),
        warmup_s=cfg_base.warmup_s,
        steady_state_s=cfg_base.steady_state_s,
        phase_timing_mode=cfg_base.phase_timing_mode,
        phase_max_requests_per_phase=cfg_base.phase_max_requests_per_phase,
        reranker_model_id=cfg_base.reranker_model_id,
        reranker_revision_tag=cfg_base.reranker_revision_tag,
        reranker_max_seq_len=cfg_base.reranker_max_seq_len,
        reranker_precision=cfg_base.reranker_precision,
        reranker_batch_size=cfg_base.reranker_batch_size,
        reranker_truncation=cfg_base.reranker_truncation,
        require_hf_backend=require_hf_backend,
        search_params=cfg_base.search_params,
    )
    original_builder = s5_mod._build_reranker_runtime
    try:
        s5_mod._build_reranker_runtime = lambda _cfg: runtime
        rows = s5_mod.run(adapter=adapter, cfg=cfg, rng=np.random.default_rng(seed))
        return rows[0]
    finally:
        s5_mod._build_reranker_runtime = original_builder


def test_s5_rerank_uses_runtime_backend_metadata() -> None:
    runtime = s5_mod._RerankerRuntime(
        backend="hf_cross_encoder",
        score_pairs=lambda _query, docs: [float(idx) for idx, _ in enumerate(docs)],
        uses_qrels_supervision=False,
        fallback_reason=None,
        device="cpu",
        local_files_only=True,
    )
    row = _run_with_runtime(runtime)
    payload = json.loads(row.info_json)
    assert payload["reranker"]["backend"] == "hf_cross_encoder"
    assert payload["reranker"]["runtime_errors"] == 0


def test_s5_rerank_counts_runtime_errors_without_crashing() -> None:
    def _raise(_query: str, _docs: list[str]) -> list[float]:
        raise RuntimeError("synthetic reranker failure")

    runtime = s5_mod._RerankerRuntime(
        backend="hf_cross_encoder",
        score_pairs=_raise,
        uses_qrels_supervision=False,
        fallback_reason=None,
        device="cpu",
        local_files_only=True,
    )
    row = _run_with_runtime(runtime, seed=11)
    payload = json.loads(row.info_json)
    assert int(payload["reranker"]["runtime_errors"]) > 0
    assert int(row.errors) >= int(payload["reranker"]["runtime_errors"])


def test_s5_require_hf_backend_fails_fast_on_proxy_runtime() -> None:
    runtime = s5_mod._RerankerRuntime(
        backend="heuristic_proxy",
        score_pairs=None,
        uses_qrels_supervision=False,
        fallback_reason="hf disabled",
        device=None,
        local_files_only=True,
    )
    adapter = _CountingMockAdapter()
    adapter.create(collection="s5-test", dimension=16, metric="ip")
    cfg = _base_cfg()
    cfg = s5_mod.S5Config(
        vector_dim=cfg.vector_dim,
        num_vectors=cfg.num_vectors,
        num_queries=cfg.num_queries,
        top_k=cfg.top_k,
        clients_read=cfg.clients_read,
        sla_threshold_ms=cfg.sla_threshold_ms,
        candidate_budgets=list(cfg.candidate_budgets),
        warmup_s=cfg.warmup_s,
        steady_state_s=cfg.steady_state_s,
        phase_timing_mode=cfg.phase_timing_mode,
        phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
        reranker_model_id=cfg.reranker_model_id,
        reranker_revision_tag=cfg.reranker_revision_tag,
        reranker_max_seq_len=cfg.reranker_max_seq_len,
        reranker_precision=cfg.reranker_precision,
        reranker_batch_size=cfg.reranker_batch_size,
        reranker_truncation=cfg.reranker_truncation,
        require_hf_backend=True,
        search_params=cfg.search_params,
    )
    original_builder = s5_mod._build_reranker_runtime
    try:
        s5_mod._build_reranker_runtime = lambda _cfg: runtime
        try:
            s5_mod.run(adapter=adapter, cfg=cfg, rng=np.random.default_rng(13))
        except RuntimeError as exc:
            assert "requires hf_cross_encoder backend" in str(exc)
            assert adapter.bulk_upsert_calls == 0
        else:
            raise AssertionError("expected RuntimeError when require_hf_backend is true and backend is heuristic")
    finally:
        s5_mod._build_reranker_runtime = original_builder
