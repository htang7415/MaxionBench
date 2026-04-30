"""Portable text-retrieval helpers shared by S1 and S3."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import numpy as np

from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset, tokenize_text, top_relevant_ids
from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import evidence_coverage_at_k, mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import sla_violation_rate
from maxionbench.scenarios.phased import run_query_phases
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class PortableTextConfig:
    top_k: int
    clients_read: int
    sla_threshold_ms: float
    warmup_s: float = 0.0
    steady_state_s: float = 0.0
    phase_timing_mode: str = "bounded"
    phase_max_requests_per_phase: int | None = None
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PortableTextResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    evidence_coverage_at_5: float
    evidence_coverage_at_10: float
    evidence_coverage_at_20: float
    avg_retrieved_input_tokens: float
    sla_violation_rate: float
    errors: int
    measured_requests: int
    measured_elapsed_s: float
    warmup_requests: int
    warmup_elapsed_s: float


def ingest_text_dataset(adapter: Any, dataset: D4RetrievalDataset) -> None:
    batch_size = 5_000
    total = len(dataset.doc_ids)
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        records = [
            UpsertRecord(
                id=dataset.doc_ids[idx],
                vector=dataset.doc_vectors[idx].tolist(),
                payload={
                    "text": dataset.doc_texts[idx],
                    "token_count": len(tokenize_text(dataset.doc_texts[idx])),
                },
            )
            for idx in range(start, stop)
        ]
        adapter.bulk_upsert(records)
    adapter.flush_or_commit()


def evaluate_text_queries(
    *,
    adapter: Any,
    cfg: PortableTextConfig,
    dataset: D4RetrievalDataset,
    observation_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> PortableTextResult:
    adapter.set_search_params(cfg.search_params or {})
    latencies_ms: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    evidence_coverage_5: list[float] = []
    evidence_coverage_10: list[float] = []
    evidence_coverage_20: list[float] = []
    retrieved_token_counts: list[float] = []
    errors = 0

    def query_once(query_idx: int) -> tuple[float, list[str], float, int]:
        qvec = dataset.query_vectors[query_idx]
        q_start = time.perf_counter()
        try:
            results = adapter.query(QueryRequest(vector=qvec.tolist(), top_k=max(cfg.top_k, 20)))
            retrieved_ids = [item.id for item in results]
            retrieved_tokens = float(
                sum(
                    int(item.payload.get("token_count", 0))
                    for item in results[: cfg.top_k]
                    if isinstance(item.payload, Mapping)
                )
            )
            err = 0
        except Exception:
            retrieved_ids = []
            retrieved_tokens = 0.0
            err = 1
        latency_ms = (time.perf_counter() - q_start) * 1000.0
        return latency_ms, retrieved_ids, retrieved_tokens, err

    measured, warmup_stats, measure_stats = run_query_phases(
        total_queries=len(dataset.query_ids),
        clients_read=cfg.clients_read,
        warmup_s=cfg.warmup_s,
        steady_state_s=cfg.steady_state_s,
        evaluate_query=query_once,
        strict_timing=cfg.phase_timing_mode == "strict",
        max_requests_per_phase=cfg.phase_max_requests_per_phase,
    )
    if not measured:
        raise RuntimeError("portable text retrieval measurement phase did not execute any query")

    for query_idx, (latency_ms, retrieved_ids, retrieved_tokens, err) in measured:
        qid = dataset.query_ids[query_idx]
        qrels = dataset.qrels[qid]
        relevant_ids = top_relevant_ids(qrels, k=max(cfg.top_k, 20))
        evidence_ids = list(qrels.keys())
        latencies_ms.append(latency_ms)
        retrieved_token_counts.append(retrieved_tokens)
        errors += err
        recall = recall_at_k(retrieved_ids, relevant_ids, k=min(10, cfg.top_k))
        ndcg = ndcg_at_10(retrieved_ids, {doc_id: float(rel) for doc_id, rel in qrels.items()})
        mrr = mrr_at_k(retrieved_ids, relevant_ids, k=min(10, cfg.top_k))
        coverage_5 = evidence_coverage_at_k(retrieved_ids, evidence_ids, k=5)
        coverage_10 = evidence_coverage_at_k(retrieved_ids, evidence_ids, k=10)
        coverage_20 = evidence_coverage_at_k(retrieved_ids, evidence_ids, k=20)
        recall_values.append(recall)
        ndcg_values.append(ndcg)
        mrr_values.append(mrr)
        evidence_coverage_5.append(coverage_5)
        evidence_coverage_10.append(coverage_10)
        evidence_coverage_20.append(coverage_20)
        if observation_sink is not None:
            observation_sink(
                {
                    "observation_type": "quality",
                    "query_index": int(query_idx),
                    "query_id": str(qid),
                    "latency_ms": float(latency_ms),
                    "retrieved_input_tokens": float(retrieved_tokens),
                    "error": int(err),
                    "recall_at_10": float(recall),
                    "ndcg_at_10": float(ndcg),
                    "mrr_at_10": float(mrr),
                    "evidence_coverage_at_5": float(coverage_5),
                    "evidence_coverage_at_10": float(coverage_10),
                    "evidence_coverage_at_20": float(coverage_20),
                }
            )

    summary = latency_summary(latencies_ms)
    over_sla = sum(1 for latency_ms in latencies_ms if latency_ms > cfg.sla_threshold_ms)
    return PortableTextResult(
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(measure_stats.requests) / max(measure_stats.elapsed_s, 1e-9),
        recall_at_10=float(np.mean(np.asarray(recall_values, dtype=np.float64))),
        ndcg_at_10=float(np.mean(np.asarray(ndcg_values, dtype=np.float64))),
        mrr_at_10=float(np.mean(np.asarray(mrr_values, dtype=np.float64))),
        evidence_coverage_at_5=float(np.mean(np.asarray(evidence_coverage_5, dtype=np.float64))),
        evidence_coverage_at_10=float(np.mean(np.asarray(evidence_coverage_10, dtype=np.float64))),
        evidence_coverage_at_20=float(np.mean(np.asarray(evidence_coverage_20, dtype=np.float64))),
        avg_retrieved_input_tokens=float(np.mean(np.asarray(retrieved_token_counts, dtype=np.float64))),
        sla_violation_rate=sla_violation_rate(
            total_requests=measure_stats.requests,
            over_sla=over_sla,
            errors=errors,
        ),
        errors=errors,
        measured_requests=measure_stats.requests,
        measured_elapsed_s=measure_stats.elapsed_s,
        warmup_requests=warmup_stats.requests,
        warmup_elapsed_s=warmup_stats.elapsed_s,
    )
