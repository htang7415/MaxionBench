"""S2 filtered ANN selectivity sweep on D3-style correlated metadata."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time
from typing import Any, Mapping

import numpy as np

from maxionbench.datasets.d3_generator import D3Dataset, D3Params, generate_d3_dataset, generate_synthetic_vectors
from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import p99_inflation, sla_violation_rate
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class S2Config:
    vector_dim: int
    num_vectors: int
    num_queries: int
    top_k: int
    clients_read: int
    sla_threshold_ms: float
    selectivities: list[float]
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class S2ConditionResult:
    selectivity: float
    filter_json: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    sla_violation_rate: float
    errors: int
    p99_inflation_vs_unfiltered: float


def run(
    adapter: Any,
    cfg: S2Config,
    rng: np.random.Generator,
    *,
    d3_params: D3Params,
    vectors: np.ndarray | None = None,
) -> list[S2ConditionResult]:
    if vectors is None:
        vectors = generate_synthetic_vectors(num_vectors=cfg.num_vectors, dim=cfg.vector_dim, seed=int(rng.integers(1, 1_000_000)))
    dataset = generate_d3_dataset(vectors, d3_params)
    _ingest_dataset(adapter, dataset)
    adapter.set_search_params(cfg.search_params or {})

    query_indices = rng.choice(dataset.vectors.shape[0], size=min(cfg.num_queries, dataset.vectors.shape[0]), replace=False)
    query_vecs = dataset.vectors[query_indices]

    ordered_selectivities = sorted(set([1.0, *cfg.selectivities]))
    raw_results: list[S2ConditionResult] = []
    baseline_p99 = 0.0
    for selectivity in ordered_selectivities:
        filt = _pick_filter_for_selectivity(dataset, selectivity=selectivity)
        result = _run_condition(
            adapter=adapter,
            dataset=dataset,
            query_vecs=query_vecs,
            clients_read=cfg.clients_read,
            top_k=cfg.top_k,
            sla_threshold_ms=cfg.sla_threshold_ms,
            filt=filt,
            selectivity=selectivity,
        )
        if abs(selectivity - 1.0) < 1e-9:
            baseline_p99 = result.p99_ms
        raw_results.append(result)

    final_results: list[S2ConditionResult] = []
    for item in raw_results:
        final_results.append(
            S2ConditionResult(
                selectivity=item.selectivity,
                filter_json=item.filter_json,
                p50_ms=item.p50_ms,
                p95_ms=item.p95_ms,
                p99_ms=item.p99_ms,
                qps=item.qps,
                recall_at_10=item.recall_at_10,
                ndcg_at_10=item.ndcg_at_10,
                mrr_at_10=item.mrr_at_10,
                sla_violation_rate=item.sla_violation_rate,
                errors=item.errors,
                p99_inflation_vs_unfiltered=p99_inflation(item.p99_ms, baseline_p99),
            )
        )
    return final_results


def _ingest_dataset(adapter: Any, dataset: D3Dataset) -> None:
    records = [
        UpsertRecord(id=dataset.ids[i], vector=dataset.vectors[i].tolist(), payload=dataset.payloads[i])
        for i in range(dataset.vectors.shape[0])
    ]
    adapter.bulk_upsert(records)
    adapter.flush_or_commit()


def _run_condition(
    *,
    adapter: Any,
    dataset: D3Dataset,
    query_vecs: np.ndarray,
    clients_read: int,
    top_k: int,
    sla_threshold_ms: float,
    filt: Mapping[str, Any] | None,
    selectivity: float,
) -> S2ConditionResult:
    latencies_ms: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    errors = 0

    def query_once(qv: np.ndarray) -> tuple[float, list[str], int]:
        request = QueryRequest(vector=qv.tolist(), top_k=top_k, filters=filt)
        start = time.perf_counter()
        try:
            rows = adapter.query(request)
            retrieved = [row.id for row in rows]
            err = 0
        except Exception:
            retrieved = []
            err = 1
        return (time.perf_counter() - start) * 1000.0, retrieved, err

    t0 = time.perf_counter()
    if clients_read <= 1:
        outputs = [query_once(qv) for qv in query_vecs]
    else:
        with ThreadPoolExecutor(max_workers=clients_read) as pool:
            outputs = list(pool.map(query_once, list(query_vecs)))
    elapsed = max(time.perf_counter() - t0, 1e-9)

    for qv, (lat, retrieved, err) in zip(query_vecs, outputs):
        latencies_ms.append(lat)
        errors += err
        gt = _exact_filtered_topk(dataset, query_vec=qv, top_k=top_k, filt=filt)
        rel = {doc_id: float(top_k - rank) for rank, doc_id in enumerate(gt)}
        recalls.append(recall_at_k(retrieved, gt, k=min(10, top_k)))
        ndcgs.append(ndcg_at_10(retrieved, rel))
        mrrs.append(mrr_at_k(retrieved, gt, k=min(10, top_k)))

    summary = latency_summary(latencies_ms)
    over_sla = sum(1 for value in latencies_ms if value > sla_threshold_ms)
    return S2ConditionResult(
        selectivity=selectivity,
        filter_json=json.dumps(filt or {"type": "unfiltered"}, sort_keys=True),
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(len(query_vecs)) / elapsed,
        recall_at_10=float(np.mean(np.asarray(recalls, dtype=np.float64))) if recalls else 0.0,
        ndcg_at_10=float(np.mean(np.asarray(ndcgs, dtype=np.float64))) if ndcgs else 0.0,
        mrr_at_10=float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else 0.0,
        sla_violation_rate=sla_violation_rate(total_requests=len(query_vecs), over_sla=over_sla, errors=errors),
        errors=errors,
        p99_inflation_vs_unfiltered=0.0,
    )


def _exact_filtered_topk(
    dataset: D3Dataset,
    *,
    query_vec: np.ndarray,
    top_k: int,
    filt: Mapping[str, Any] | None,
) -> list[str]:
    if not filt:
        idx = np.arange(dataset.vectors.shape[0])
    else:
        idx = _filter_indices(dataset, filt)
    if idx.size == 0:
        return []
    scores = dataset.vectors[idx] @ query_vec
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [dataset.ids[int(idx[o])] for o in order]


def _filter_indices(dataset: D3Dataset, filt: Mapping[str, Any]) -> np.ndarray:
    if "tenant_id" in filt:
        tenant_str = str(filt["tenant_id"])
        tenant_num = int(tenant_str.split("-")[-1])
        return np.where(dataset.tenant_ids == tenant_num)[0]
    if "acl_bucket" in filt:
        return np.where(dataset.acl_buckets == int(filt["acl_bucket"]))[0]
    if "time_bucket" in filt:
        return np.where(dataset.time_buckets == int(filt["time_bucket"]))[0]
    return np.arange(dataset.vectors.shape[0])


def _pick_filter_for_selectivity(dataset: D3Dataset, *, selectivity: float) -> Mapping[str, Any] | None:
    if selectivity >= 0.999:
        return None
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    n = float(dataset.vectors.shape[0])

    for tenant in range(dataset.params.num_tenants):
        ratio = float(np.count_nonzero(dataset.tenant_ids == tenant)) / n
        candidates.append((abs(ratio - selectivity), {"tenant_id": f"tenant-{tenant:03d}"}))
    for acl in range(dataset.params.num_acl_buckets):
        ratio = float(np.count_nonzero(dataset.acl_buckets == acl)) / n
        candidates.append((abs(ratio - selectivity), {"acl_bucket": acl}))
    for bucket in range(dataset.params.num_time_buckets):
        ratio = float(np.count_nonzero(dataset.time_buckets == bucket)) / n
        candidates.append((abs(ratio - selectivity), {"time_bucket": bucket}))

    candidates.sort(key=lambda item: item[0])
    return dict(candidates[0][1])
