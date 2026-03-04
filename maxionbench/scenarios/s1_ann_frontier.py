"""S1 Pure ANN frontier scenario (deterministic synthetic harness baseline)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import sla_violation_rate
from maxionbench.scenarios.phased import run_query_phases
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class S1Config:
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
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class S1Data:
    ids: list[str]
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: list[list[str]] | None = None


@dataclass(frozen=True)
class S1Result:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    sla_violation_rate: float
    errors: int
    measured_requests: int
    measured_elapsed_s: float
    warmup_requests: int
    warmup_elapsed_s: float


def _exact_topk_ids(
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int,
) -> list[int]:
    scores = vectors @ query
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return indices.tolist()


def run(adapter: Any, cfg: S1Config, rng: np.random.Generator) -> S1Result:
    return run_with_data(adapter=adapter, cfg=cfg, rng=rng, data=None)


def run_with_data(
    adapter: Any,
    cfg: S1Config,
    rng: np.random.Generator,
    data: S1Data | None,
) -> S1Result:
    if data is None:
        ids = [f"doc-{idx:07d}" for idx in range(cfg.num_vectors)]
        vectors = rng.standard_normal((cfg.num_vectors, cfg.vector_dim), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        queries = vectors[rng.choice(cfg.num_vectors, size=cfg.num_queries, replace=False)]
        gt_ids_by_query: list[list[str]] | None = None
    else:
        ids = list(data.ids)
        vectors = np.asarray(data.vectors, dtype=np.float32)
        queries = np.asarray(data.queries, dtype=np.float32)[: cfg.num_queries]
        gt_ids_by_query = data.ground_truth_ids[: cfg.num_queries] if data.ground_truth_ids else None

    payloads = [{"tenant_id": f"tenant-{idx % 100:03d}", "acl_bucket": idx % 16} for idx in range(vectors.shape[0])]
    records = [UpsertRecord(id=doc_id, vector=vectors[idx].tolist(), payload=payloads[idx]) for idx, doc_id in enumerate(ids)]

    adapter.bulk_upsert(records)
    adapter.flush_or_commit()
    adapter.set_search_params(cfg.search_params or {})

    query_vectors = queries
    num_queries = int(query_vectors.shape[0])
    if num_queries < 1:
        raise ValueError("S1 requires at least one query.")

    latencies_ms: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    errors = 0

    gt_rows = _ground_truth_rows(
        vectors=vectors,
        ids=ids,
        query_vectors=query_vectors,
        top_k=cfg.top_k,
        precomputed=gt_ids_by_query,
    )
    relevance_rows = [{doc_id: float(cfg.top_k - rank) for rank, doc_id in enumerate(gt_ids)} for gt_ids in gt_rows]

    def query_once(query_vec: np.ndarray) -> tuple[float, list[str], int]:
        req = QueryRequest(vector=query_vec.tolist(), top_k=cfg.top_k)
        q_start = time.perf_counter()
        try:
            results = adapter.query(req)
            retrieved_ids = [item.id for item in results]
            err = 0
        except Exception:
            retrieved_ids = []
            err = 1
        latency_ms = (time.perf_counter() - q_start) * 1000.0
        return latency_ms, retrieved_ids, err

    measured, warmup_stats, measure_stats = run_query_phases(
        total_queries=num_queries,
        clients_read=cfg.clients_read,
        warmup_s=cfg.warmup_s,
        steady_state_s=cfg.steady_state_s,
        evaluate_query=lambda idx: query_once(query_vectors[idx]),
        strict_timing=cfg.phase_timing_mode == "strict",
        max_requests_per_phase=cfg.phase_max_requests_per_phase,
    )
    if not measured:
        raise RuntimeError("S1 measurement phase did not execute any query")

    for index, (latency_ms, retrieved_ids, err) in measured:
        latencies_ms.append(latency_ms)
        errors += err
        gt_ids = gt_rows[index]
        relevance = relevance_rows[index]
        recall_values.append(recall_at_k(retrieved_ids, gt_ids, k=min(10, cfg.top_k)))
        ndcg_values.append(ndcg_at_10(retrieved_ids, relevance))
        mrr_values.append(mrr_at_k(retrieved_ids, gt_ids, k=min(10, cfg.top_k)))

    summary = latency_summary(latencies_ms)
    over_sla = sum(1 for value in latencies_ms if value > cfg.sla_threshold_ms)

    return S1Result(
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=measure_stats.requests / max(measure_stats.elapsed_s, 1e-9),
        recall_at_10=float(np.mean(recall_values)),
        ndcg_at_10=float(np.mean(ndcg_values)),
        mrr_at_10=float(np.mean(mrr_values)),
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


def _ground_truth_rows(
    *,
    vectors: np.ndarray,
    ids: list[str],
    query_vectors: np.ndarray,
    top_k: int,
    precomputed: list[list[str]] | None,
) -> list[list[str]]:
    if precomputed is not None:
        return [list(row[:top_k]) for row in precomputed[: len(query_vectors)]]
    rows: list[list[str]] = []
    for query_vec in query_vectors:
        gt_indices = _exact_topk_ids(vectors=vectors, query=query_vec, top_k=top_k)
        rows.append([ids[i] for i in gt_indices])
    return rows
