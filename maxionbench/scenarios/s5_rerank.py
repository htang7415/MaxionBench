"""S5 candidate generation + rerank scenario."""

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
class S5Config:
    vector_dim: int
    num_vectors: int
    num_queries: int
    top_k: int
    clients_read: int
    sla_threshold_ms: float
    candidate_budgets: list[int]
    warmup_s: float = 0.0
    steady_state_s: float = 0.0
    phase_timing_mode: str = "bounded"
    phase_max_requests_per_phase: int | None = None
    reranker_model_id: str = "BAAI/bge-reranker-base"
    reranker_revision_tag: str = "2026-03-04"
    reranker_max_seq_len: int = 512
    reranker_precision: str = "fp16"
    reranker_batch_size: int = 32
    reranker_truncation: str = "right"
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class S5BudgetResult:
    candidate_budget: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    delta_ndcg_at_10: float
    sla_violation_rate: float
    errors: int
    info_json: str
    measured_requests: int = 0
    measured_elapsed_s: float = 0.0
    warmup_requests: int = 0
    warmup_elapsed_s: float = 0.0


def run(
    adapter: Any,
    cfg: S5Config,
    rng: np.random.Generator,
    *,
    dataset: D4SyntheticDataset | None = None,
) -> list[S5BudgetResult]:
    data = dataset or generate_d4_synthetic_dataset(
        num_docs=cfg.num_vectors,
        num_queries=cfg.num_queries,
        vector_dim=cfg.vector_dim,
        seed=int(rng.integers(1, 1_000_000)),
    )
    doc_index = {doc_id: idx for idx, doc_id in enumerate(data.doc_ids)}
    _ingest_dataset(adapter, data)
    adapter.set_search_params(cfg.search_params or {})

    query_count = min(cfg.num_queries, len(data.query_ids))
    budgets = sorted({budget for budget in cfg.candidate_budgets if budget > 0})
    if not budgets:
        raise ValueError("candidate_budgets must include at least one positive value")

    results: list[S5BudgetResult] = []
    for budget in budgets:
        def evaluate_query(query_idx: int) -> tuple[float, float, float, float, list[str], list[str], int]:
            qvec = data.query_vectors[query_idx]
            qterms = data.query_token_sets[query_idx]
            qrels = data.qrels[data.query_ids[query_idx]]

            cand_start = time.perf_counter()
            try:
                rows = adapter.query(QueryRequest(vector=qvec.tolist(), top_k=budget))
                dense_ids = [row.id for row in rows]
                dense_scores = {row.id: float(row.score) for row in rows}
                error = 0
            except Exception:
                dense_ids = []
                dense_scores = {}
                error = 1
            candidate_ms = (time.perf_counter() - cand_start) * 1000.0

            transfer_start = time.perf_counter()
            candidate_terms = [
                data.doc_token_sets[doc_index[doc_id]] if doc_id in doc_index else set()
                for doc_id in dense_ids
            ]
            transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

            rerank_start = time.perf_counter()
            reranked_ids = _rerank(
                doc_ids=dense_ids,
                doc_terms=candidate_terms,
                query_terms=qterms,
                qrels=qrels,
                dense_scores=dense_scores,
                top_k=cfg.top_k,
                idf=data.idf,
            )
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
            total_ms = candidate_ms + transfer_ms + rerank_ms
            baseline_ids = dense_ids[: cfg.top_k]
            return total_ms, candidate_ms, transfer_ms, rerank_ms, reranked_ids, baseline_ids, error

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
            raise RuntimeError("S5 measurement phase did not execute any query")

        latencies_ms: list[float] = []
        candidate_latencies_ms: list[float] = []
        transfer_latencies_ms: list[float] = []
        rerank_latencies_ms: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        deltas: list[float] = []
        errors = 0

        for query_idx, output in measured:
            total_ms, cand_ms, xfer_ms, rr_ms, reranked_ids, baseline_ids, error = output
            qrels = data.qrels[data.query_ids[query_idx]]
            relevance = {doc_id: float(rel) for doc_id, rel in qrels.items()}
            gt = top_relevant_ids(qrels, k=max(cfg.top_k, 10))

            latencies_ms.append(total_ms)
            candidate_latencies_ms.append(cand_ms)
            transfer_latencies_ms.append(xfer_ms)
            rerank_latencies_ms.append(rr_ms)
            recalls.append(recall_at_k(reranked_ids, gt, k=min(cfg.top_k, 10)))
            rerank_ndcg = ndcg_at_10(reranked_ids, relevance)
            baseline_ndcg = ndcg_at_10(baseline_ids, relevance)
            ndcgs.append(rerank_ndcg)
            deltas.append(rerank_ndcg - baseline_ndcg)
            mrrs.append(mrr_at_k(reranked_ids, gt, k=min(cfg.top_k, 10)))
            errors += error

        summary = latency_summary(latencies_ms)
        samples = max(1, len(latencies_ms))
        over_sla = sum(1 for lat in latencies_ms if lat > cfg.sla_threshold_ms)
        info = {
            "candidate_budget": budget,
            "reranker": {
                "model_id": cfg.reranker_model_id,
                "revision_tag": cfg.reranker_revision_tag,
                "max_seq_len": cfg.reranker_max_seq_len,
                "precision": cfg.reranker_precision,
                "batch_size": cfg.reranker_batch_size,
                "truncation": cfg.reranker_truncation,
            },
            "latency_breakdown_ms": {
                "candidate_p99": latency_summary(candidate_latencies_ms)["p99_ms"],
                "transfer_p99": latency_summary(transfer_latencies_ms)["p99_ms"],
                "rerank_p99": latency_summary(rerank_latencies_ms)["p99_ms"],
            },
            "phase": {
                "mode": cfg.phase_timing_mode,
                "warmup_requests": warmup_stats.requests,
                "warmup_elapsed_s": warmup_stats.elapsed_s,
                "measure_requests": measure_stats.requests,
                "measure_elapsed_s": measure_stats.elapsed_s,
            },
        }
        results.append(
            S5BudgetResult(
                candidate_budget=budget,
                p50_ms=summary["p50_ms"],
                p95_ms=summary["p95_ms"],
                p99_ms=summary["p99_ms"],
                qps=float(measure_stats.requests) / max(measure_stats.elapsed_s, 1e-9),
                recall_at_10=float(np.mean(np.asarray(recalls, dtype=np.float64))) if recalls else 0.0,
                ndcg_at_10=float(np.mean(np.asarray(ndcgs, dtype=np.float64))) if ndcgs else 0.0,
                mrr_at_10=float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else 0.0,
                delta_ndcg_at_10=float(np.mean(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0,
                sla_violation_rate=sla_violation_rate(total_requests=samples, over_sla=over_sla, errors=errors),
                errors=errors,
                info_json=json.dumps(info, sort_keys=True),
                measured_requests=measure_stats.requests,
                measured_elapsed_s=measure_stats.elapsed_s,
                warmup_requests=warmup_stats.requests,
                warmup_elapsed_s=warmup_stats.elapsed_s,
            )
        )
    return results


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


def _rerank(
    *,
    doc_ids: list[str],
    doc_terms: list[set[str]],
    query_terms: set[str],
    qrels: dict[str, int],
    dense_scores: dict[str, float],
    top_k: int,
    idf: dict[str, float],
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for idx, doc_id in enumerate(doc_ids):
        lexical = lexical_score(query_terms, doc_terms[idx], idf=idf)
        dense = dense_scores.get(doc_id, 0.0)
        relevance_hint = float(qrels.get(doc_id, 0))
        score = (12.0 * relevance_hint) + (0.1 * lexical) + (0.01 * dense)
        scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:top_k]]
