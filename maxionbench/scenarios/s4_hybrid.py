"""S4 hybrid retrieval scenario with dense baseline and BM25+dense RRF fusion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class S4Config:
    vector_dim: int
    num_vectors: int
    num_queries: int
    top_k: int
    clients_read: int
    sla_threshold_ms: float
    dense_candidates: int = 200
    bm25_candidates: int = 200
    rrf_k: int = 60
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class S4ConditionResult:
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


def run(
    adapter: Any,
    cfg: S4Config,
    rng: np.random.Generator,
    *,
    dataset: D4SyntheticDataset | None = None,
) -> list[S4ConditionResult]:
    data = dataset or generate_d4_synthetic_dataset(
        num_docs=cfg.num_vectors,
        num_queries=cfg.num_queries,
        vector_dim=cfg.vector_dim,
        seed=int(rng.integers(1, 1_000_000)),
    )
    _ingest_dataset(adapter, data)
    adapter.set_search_params(cfg.search_params or {})

    query_count = min(cfg.num_queries, len(data.query_ids))
    indices = list(range(query_count))

    def evaluate_query(query_idx: int) -> tuple[float, list[str], int, float, list[str], int]:
        qvec = data.query_vectors[query_idx]
        qterms = data.query_token_sets[query_idx]

        dense_start = time.perf_counter()
        try:
            dense_rows = adapter.query(QueryRequest(vector=qvec.tolist(), top_k=cfg.dense_candidates))
            dense_ids = [row.id for row in dense_rows]
            dense_error = 0
        except Exception:
            dense_ids = []
            dense_error = 1
        dense_ms = (time.perf_counter() - dense_start) * 1000.0

        bm25_start = time.perf_counter()
        bm25_ids = _bm25_topk(data, query_terms=qterms, top_k=cfg.bm25_candidates)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000.0

        fuse_start = time.perf_counter()
        hybrid_ids = _rrf_fuse(
            rankings=[dense_ids[: cfg.dense_candidates], bm25_ids],
            rrf_k=cfg.rrf_k,
            top_k=cfg.top_k,
        )
        fuse_ms = (time.perf_counter() - fuse_start) * 1000.0
        hybrid_ms = dense_ms + bm25_ms + fuse_ms
        hybrid_error = dense_error
        return dense_ms, dense_ids[: cfg.top_k], dense_error, hybrid_ms, hybrid_ids, hybrid_error

    if cfg.clients_read <= 1:
        outputs = [evaluate_query(i) for i in indices]
    else:
        with ThreadPoolExecutor(max_workers=cfg.clients_read) as pool:
            outputs = list(pool.map(evaluate_query, indices))

    dense_lats: list[float] = []
    dense_recalls: list[float] = []
    dense_ndcgs: list[float] = []
    dense_mrrs: list[float] = []
    dense_errors = 0

    hybrid_lats: list[float] = []
    hybrid_recalls: list[float] = []
    hybrid_ndcgs: list[float] = []
    hybrid_mrrs: list[float] = []
    hybrid_errors = 0

    for query_idx, output in enumerate(outputs):
        dense_ms, dense_ids, dense_error, hybrid_ms, hybrid_ids, hybrid_error = output
        qrels = data.qrels[data.query_ids[query_idx]]
        gt = top_relevant_ids(qrels, k=max(cfg.top_k, 10))
        relevance = {doc_id: float(rel) for doc_id, rel in qrels.items()}

        dense_lats.append(dense_ms)
        dense_errors += dense_error
        dense_recalls.append(recall_at_k(dense_ids, gt, k=min(cfg.top_k, 10)))
        dense_ndcgs.append(ndcg_at_10(dense_ids, relevance))
        dense_mrrs.append(mrr_at_k(dense_ids, gt, k=min(cfg.top_k, 10)))

        hybrid_lats.append(hybrid_ms)
        hybrid_errors += hybrid_error
        hybrid_recalls.append(recall_at_k(hybrid_ids, gt, k=min(cfg.top_k, 10)))
        hybrid_ndcgs.append(ndcg_at_10(hybrid_ids, relevance))
        hybrid_mrrs.append(mrr_at_k(hybrid_ids, gt, k=min(cfg.top_k, 10)))

    dense = _aggregate(
        mode="dense_only",
        latencies_ms=dense_lats,
        recalls=dense_recalls,
        ndcgs=dense_ndcgs,
        mrrs=dense_mrrs,
        errors=dense_errors,
        sla_threshold_ms=cfg.sla_threshold_ms,
        dense_candidates=cfg.dense_candidates,
        bm25_candidates=cfg.bm25_candidates,
        rrf_k=cfg.rrf_k,
    )
    hybrid = _aggregate(
        mode="bm25_dense_rrf",
        latencies_ms=hybrid_lats,
        recalls=hybrid_recalls,
        ndcgs=hybrid_ndcgs,
        mrrs=hybrid_mrrs,
        errors=hybrid_errors,
        sla_threshold_ms=cfg.sla_threshold_ms,
        dense_candidates=cfg.dense_candidates,
        bm25_candidates=cfg.bm25_candidates,
        rrf_k=cfg.rrf_k,
    )
    return [dense, hybrid]


def _aggregate(
    *,
    mode: str,
    latencies_ms: list[float],
    recalls: list[float],
    ndcgs: list[float],
    mrrs: list[float],
    errors: int,
    sla_threshold_ms: float,
    dense_candidates: int,
    bm25_candidates: int,
    rrf_k: int,
) -> S4ConditionResult:
    samples = max(1, len(latencies_ms))
    summary = latency_summary(latencies_ms)
    elapsed_s = max(sum(latencies_ms) / 1000.0, 1e-9)
    info = {
        "mode": mode,
        "dense_candidates": dense_candidates,
        "bm25_candidates": bm25_candidates,
        "k_rrf": rrf_k,
    }
    over_sla = sum(1 for lat in latencies_ms if lat > sla_threshold_ms)
    return S4ConditionResult(
        mode=mode,
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(samples) / elapsed_s,
        recall_at_10=float(np.mean(np.asarray(recalls, dtype=np.float64))) if recalls else 0.0,
        ndcg_at_10=float(np.mean(np.asarray(ndcgs, dtype=np.float64))) if ndcgs else 0.0,
        mrr_at_10=float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else 0.0,
        sla_violation_rate=sla_violation_rate(total_requests=samples, over_sla=over_sla, errors=errors),
        errors=errors,
        info_json=json.dumps(info, sort_keys=True),
    )


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
