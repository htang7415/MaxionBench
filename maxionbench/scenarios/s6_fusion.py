"""S6 multi-index fusion scenario."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping

import numpy as np

from maxionbench.datasets.loaders.d4_synthetic import (
    D4SyntheticDataset,
    generate_d4_synthetic_dataset,
    lexical_score,
    top_relevant_ids,
)
from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import sla_violation_rate
from maxionbench.scenarios.phased import run_query_phases
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class S6Config:
    vector_dim: int
    num_vectors: int
    num_queries: int
    top_k: int
    clients_read: int
    sla_threshold_ms: float
    warmup_s: float = 0.0
    steady_state_s: float = 0.0
    phase_timing_mode: str = "bounded"
    phase_max_requests_per_phase: int | None = None
    rrf_k: int = 60
    dense_a_candidates: int = 200
    dense_b_candidates: int = 200
    bm25_candidates: int = 200
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class S6ConditionResult:
    mode: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    sla_violation_rate: float
    errors: int
    info_json: str
    measured_requests: int = 0
    measured_elapsed_s: float = 0.0
    warmup_requests: int = 0
    warmup_elapsed_s: float = 0.0


def run(
    adapter: Any,
    cfg: S6Config,
    rng: np.random.Generator,
    *,
    dataset: D4SyntheticDataset | None = None,
) -> list[S6ConditionResult]:
    data = dataset or generate_d4_synthetic_dataset(
        num_docs=cfg.num_vectors,
        num_queries=cfg.num_queries,
        vector_dim=cfg.vector_dim,
        seed=int(rng.integers(1, 1_000_000)),
    )
    _ingest_dataset(adapter, data)
    adapter.set_search_params(cfg.search_params or {})

    query_count = min(cfg.num_queries, len(data.query_ids))
    alt_queries = _build_alt_queries(data.query_vectors[:query_count], rng=rng)

    def evaluate_query(query_idx: int) -> tuple[float, list[str], int, float, list[str], int]:
        qvec = data.query_vectors[query_idx]
        qvec_alt = alt_queries[query_idx]
        qterms = data.query_token_sets[query_idx]

        dense_a_start = time.perf_counter()
        try:
            dense_a_rows = adapter.query(QueryRequest(vector=qvec.tolist(), top_k=cfg.dense_a_candidates))
            dense_a_ids = [row.id for row in dense_a_rows]
            err_a = 0
        except Exception:
            dense_a_ids = []
            err_a = 1
        dense_a_ms = (time.perf_counter() - dense_a_start) * 1000.0

        dense_b_start = time.perf_counter()
        try:
            dense_b_rows = adapter.query(QueryRequest(vector=qvec_alt.tolist(), top_k=cfg.dense_b_candidates))
            dense_b_ids = [row.id for row in dense_b_rows]
            err_b = 0
        except Exception:
            dense_b_ids = []
            err_b = 1
        dense_b_ms = (time.perf_counter() - dense_b_start) * 1000.0

        bm25_start = time.perf_counter()
        bm25_ids = _bm25_topk(data, query_terms=qterms, top_k=cfg.bm25_candidates)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000.0

        fuse_a_start = time.perf_counter()
        s6a_ids = _rrf_fuse(
            rankings=[dense_a_ids[: cfg.dense_a_candidates], dense_b_ids[: cfg.dense_b_candidates]],
            rrf_k=cfg.rrf_k,
            top_k=cfg.top_k,
        )
        fuse_a_ms = (time.perf_counter() - fuse_a_start) * 1000.0

        fuse_b_start = time.perf_counter()
        s6b_ids = _rrf_fuse(
            rankings=[dense_a_ids[: cfg.dense_a_candidates], bm25_ids],
            rrf_k=cfg.rrf_k,
            top_k=cfg.top_k,
        )
        fuse_b_ms = (time.perf_counter() - fuse_b_start) * 1000.0

        s6a_ms = dense_a_ms + dense_b_ms + fuse_a_ms
        s6b_ms = dense_a_ms + bm25_ms + fuse_b_ms
        return s6a_ms, s6a_ids, err_a + err_b, s6b_ms, s6b_ids, err_a

    measured, warmup_stats, measure_stats = run_query_phases(
        total_queries=query_count,
        clients_read=cfg.clients_read,
        warmup_s=cfg.warmup_s,
        steady_state_s=cfg.steady_state_s,
        evaluate_query=evaluate_query,
        strict_timing=cfg.phase_timing_mode == "strict",
        max_requests_per_phase=cfg.phase_max_requests_per_phase,
    )
    if not measured:
        raise RuntimeError("S6 measurement phase did not execute any query")

    s6a_lats: list[float] = []
    s6a_recalls: list[float] = []
    s6a_ndcgs: list[float] = []
    s6a_mrrs: list[float] = []
    s6a_errors = 0

    s6b_lats: list[float] = []
    s6b_recalls: list[float] = []
    s6b_ndcgs: list[float] = []
    s6b_mrrs: list[float] = []
    s6b_errors = 0

    for query_idx, output in measured:
        s6a_ms, s6a_ids, s6a_error, s6b_ms, s6b_ids, s6b_error = output
        qrels = data.qrels[data.query_ids[query_idx]]
        gt = top_relevant_ids(qrels, k=max(cfg.top_k, 10))
        relevance = {doc_id: float(rel) for doc_id, rel in qrels.items()}

        s6a_lats.append(s6a_ms)
        s6a_errors += s6a_error
        s6a_recalls.append(recall_at_k(s6a_ids, gt, k=min(cfg.top_k, 10)))
        s6a_ndcgs.append(ndcg_at_10(s6a_ids, relevance))
        s6a_mrrs.append(mrr_at_k(s6a_ids, gt, k=min(cfg.top_k, 10)))

        s6b_lats.append(s6b_ms)
        s6b_errors += s6b_error
        s6b_recalls.append(recall_at_k(s6b_ids, gt, k=min(cfg.top_k, 10)))
        s6b_ndcgs.append(ndcg_at_10(s6b_ids, relevance))
        s6b_mrrs.append(mrr_at_k(s6b_ids, gt, k=min(cfg.top_k, 10)))

    s6a = _aggregate(
        mode="s6a_dense_dense_rrf",
        latencies_ms=s6a_lats,
        recalls=s6a_recalls,
        ndcgs=s6a_ndcgs,
        mrrs=s6a_mrrs,
        errors=s6a_errors,
        sla_threshold_ms=cfg.sla_threshold_ms,
        cfg=cfg,
        measured_requests=measure_stats.requests,
        measured_elapsed_s=measure_stats.elapsed_s,
        warmup_requests=warmup_stats.requests,
        warmup_elapsed_s=warmup_stats.elapsed_s,
    )
    s6b = _aggregate(
        mode="s6b_dense_bm25_rrf",
        latencies_ms=s6b_lats,
        recalls=s6b_recalls,
        ndcgs=s6b_ndcgs,
        mrrs=s6b_mrrs,
        errors=s6b_errors,
        sla_threshold_ms=cfg.sla_threshold_ms,
        cfg=cfg,
        measured_requests=measure_stats.requests,
        measured_elapsed_s=measure_stats.elapsed_s,
        warmup_requests=warmup_stats.requests,
        warmup_elapsed_s=warmup_stats.elapsed_s,
    )
    return [s6a, s6b]


def _aggregate(
    *,
    mode: str,
    latencies_ms: list[float],
    recalls: list[float],
    ndcgs: list[float],
    mrrs: list[float],
    errors: int,
    sla_threshold_ms: float,
    cfg: S6Config,
    measured_requests: int,
    measured_elapsed_s: float,
    warmup_requests: int,
    warmup_elapsed_s: float,
) -> S6ConditionResult:
    samples = max(1, len(latencies_ms))
    summary = latency_summary(latencies_ms)
    over_sla = sum(1 for lat in latencies_ms if lat > sla_threshold_ms)
    info = {
        "mode": mode,
        "k_rrf": cfg.rrf_k,
        "dense_a_candidates": cfg.dense_a_candidates,
        "dense_b_candidates": cfg.dense_b_candidates,
        "bm25_candidates": cfg.bm25_candidates,
        "phase": {
            "mode": cfg.phase_timing_mode,
            "warmup_requests": warmup_requests,
            "warmup_elapsed_s": warmup_elapsed_s,
            "measure_requests": measured_requests,
            "measure_elapsed_s": measured_elapsed_s,
        },
    }
    return S6ConditionResult(
        mode=mode,
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(measured_requests) / max(measured_elapsed_s, 1e-9),
        recall_at_10=float(np.mean(np.asarray(recalls, dtype=np.float64))) if recalls else 0.0,
        ndcg_at_10=float(np.mean(np.asarray(ndcgs, dtype=np.float64))) if ndcgs else 0.0,
        mrr_at_10=float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else 0.0,
        sla_violation_rate=sla_violation_rate(total_requests=samples, over_sla=over_sla, errors=errors),
        errors=errors,
        info_json=json.dumps(info, sort_keys=True),
        measured_requests=measured_requests,
        measured_elapsed_s=measured_elapsed_s,
        warmup_requests=warmup_requests,
        warmup_elapsed_s=warmup_elapsed_s,
    )


def _build_alt_queries(queries: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(queries.shape, dtype=np.float32)
    mixed = (0.9 * queries) + (0.1 * noise)
    mixed /= np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-12
    return mixed.astype(np.float32, copy=False)


def _ingest_dataset(adapter: Any, dataset: D4SyntheticDataset) -> None:
    records = [
        UpsertRecord(
            id=doc_id,
            vector=dataset.doc_vectors[idx].tolist(),
            payload={"text": dataset.doc_texts[idx]},
        )
        for idx, doc_id in enumerate(dataset.doc_ids)
    ]
    adapter.bulk_upsert(records)
    adapter.flush_or_commit()


def _bm25_topk(dataset: D4SyntheticDataset, *, query_terms: set[str], top_k: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for doc_id, doc_terms in zip(dataset.doc_ids, dataset.doc_token_sets):
        score = lexical_score(query_terms, doc_terms, idf=dataset.idf)
        if score <= 0:
            continue
        scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:top_k]]


def _rrf_fuse(*, rankings: list[list[str]], rrf_k: int, top_k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / float(rrf_k + rank))
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ordered[:top_k]]
