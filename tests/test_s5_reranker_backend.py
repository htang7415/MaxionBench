from __future__ import annotations

import json

import numpy as np

from maxionbench.adapters.mock import MockAdapter
from maxionbench.scenarios import s5_rerank as s5_mod


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


def _run_with_runtime(runtime: s5_mod._RerankerRuntime, *, seed: int = 7) -> s5_mod.S5BudgetResult:
    adapter = MockAdapter()
    adapter.create(collection="s5-test", dimension=16, metric="ip")
    cfg = _base_cfg()
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
