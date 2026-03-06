"""S3 dynamic churn smooth workload on D3-style metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable

import numpy as np

from maxionbench.datasets.d3_generator import D3Params, generate_d3_dataset, generate_synthetic_vectors
from maxionbench.metrics.latency import latency_summary
from maxionbench.metrics.quality import mrr_at_k, ndcg_at_10, recall_at_k
from maxionbench.metrics.robustness import sla_violation_rate
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


@dataclass(frozen=True)
class S3Config:
    vector_dim: int
    num_vectors: int
    num_queries: int
    top_k: int
    sla_threshold_ms: float
    warmup_s: float
    steady_state_s: float
    lambda_req_s: float
    read_rate: float
    insert_rate: float
    update_rate: float
    delete_rate: float
    maintenance_interval_s: float
    phase_timing_mode: str = "bounded"
    max_events: int = 5000


@dataclass(frozen=True)
class S3Result:
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
    measured_requests: int
    measured_elapsed_s: float
    warmup_requests: int
    warmup_elapsed_s: float


def run(
    adapter: Any,
    cfg: S3Config,
    rng: np.random.Generator,
    *,
    d3_params: D3Params,
    burst_multiplier_fn: Callable[[float], float] | None = None,
) -> S3Result:
    vectors = generate_synthetic_vectors(num_vectors=cfg.num_vectors, dim=cfg.vector_dim, seed=int(rng.integers(1, 1_000_000)))
    dataset = generate_d3_dataset(vectors, d3_params)
    state = _ChurnState.from_dataset(dataset)
    _ingest_state(adapter, state)

    warmup_events = _phase_event_count(cfg, phase_s=cfg.warmup_s)
    measure_events = _phase_event_count(cfg, phase_s=cfg.steady_state_s)
    total_rate = max(cfg.lambda_req_s, 1.0)
    next_maintenance_s = cfg.maintenance_interval_s

    def run_phase(
        *,
        event_count: int,
        collect_metrics: bool,
    ) -> tuple[list[float], list[float], list[float], list[float], int, int, float]:
        nonlocal next_maintenance_s
        latencies_ms: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        errors = 0
        reads = 0
        phase_clock_s = 0.0
        dt_s = 1.0 / total_rate
        started = time.perf_counter()
        for _ in range(event_count):
            sim_t = phase_clock_s
            if collect_metrics:
                while sim_t >= next_maintenance_s:
                    adapter.optimize_or_compact()
                    next_maintenance_s += cfg.maintenance_interval_s

            multiplier = burst_multiplier_fn(sim_t) if burst_multiplier_fn else 1.0
            op = _sample_operation(cfg, rng, write_multiplier=multiplier)
            phase_clock_s += dt_s
            if op == "read":
                reads += 1
                q_idx = int(rng.integers(0, max(1, state.vectors.shape[0])))
                qvec = state.vectors[q_idx]
                exact = state.exact_topk(qvec, cfg.top_k)
                t0 = time.perf_counter()
                try:
                    rows = adapter.query(QueryRequest(vector=qvec.tolist(), top_k=cfg.top_k))
                    got = [row.id for row in rows]
                except Exception:
                    got = []
                    errors += 1
                if collect_metrics:
                    k_eval = min(10, cfg.top_k)
                    relevance = _ann_relevance_from_exact(exact_ids=exact, k=k_eval)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    recalls.append(recall_at_k(got, exact, k=k_eval))
                    ndcgs.append(ndcg_at_10(got, relevance))
                    mrrs.append(mrr_at_k(got, exact, k=k_eval))
                continue

            if op == "insert":
                state.insert_one(rng)
                record = state.latest_record()
                adapter.insert(record)
                adapter.flush_or_commit()
            elif op == "update":
                target = state.sample_id(rng)
                if target is not None:
                    new_vec = _random_unit_vector(cfg.vector_dim, rng)
                    state.update_vector(target, new_vec)
                    adapter.update_vectors([target], [new_vec.tolist()])
                    adapter.flush_or_commit()
            elif op == "delete":
                target = state.sample_id(rng)
                if target is not None:
                    state.delete_id(target)
                    adapter.delete([target])
                    adapter.flush_or_commit()
        elapsed_s = time.perf_counter() - started
        return latencies_ms, recalls, ndcgs, mrrs, errors, reads, elapsed_s

    _, _, _, _, _, _, warmup_elapsed = run_phase(event_count=warmup_events, collect_metrics=False)
    latencies_ms, recalls, ndcgs, mrrs, errors, reads, measure_elapsed = run_phase(
        event_count=measure_events,
        collect_metrics=True,
    )

    elapsed = max(measure_elapsed, 1e-9)
    summary = latency_summary(latencies_ms)
    over_sla = sum(1 for x in latencies_ms if x > cfg.sla_threshold_ms)
    info = {
        "mode": "s3_smooth",
        "events": measure_events,
        "reads": reads,
        "lambda_req_s": cfg.lambda_req_s,
        "read_rate": cfg.read_rate,
        "insert_rate": cfg.insert_rate,
        "update_rate": cfg.update_rate,
        "delete_rate": cfg.delete_rate,
        "burst_clock_anchor": "measurement_start",
        "phase": {
            "warmup_events": warmup_events,
            "warmup_elapsed_s": warmup_elapsed,
            "measure_events": measure_events,
            "measure_elapsed_s": measure_elapsed,
        },
    }
    mean_recall = float(np.mean(np.asarray(recalls, dtype=np.float64))) if recalls else 0.0
    mean_ndcg = float(np.mean(np.asarray(ndcgs, dtype=np.float64))) if ndcgs else 0.0
    mean_mrr = float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else 0.0
    return S3Result(
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(reads) / elapsed,
        recall_at_10=mean_recall,
        ndcg_at_10=mean_ndcg,
        mrr_at_10=mean_mrr,
        sla_violation_rate=sla_violation_rate(total_requests=max(1, reads), over_sla=over_sla, errors=errors),
        errors=errors,
        info_json=json.dumps(info, sort_keys=True),
        measured_requests=reads,
        measured_elapsed_s=measure_elapsed,
        warmup_requests=warmup_events,
        warmup_elapsed_s=warmup_elapsed,
    )


def _phase_event_count(cfg: S3Config, *, phase_s: float) -> int:
    if phase_s <= 0:
        base = max(1, cfg.num_queries)
    elif cfg.phase_timing_mode == "strict":
        base = max(1, int(cfg.lambda_req_s * phase_s))
    else:
        base = max(cfg.num_queries, int(cfg.lambda_req_s * phase_s))
    return int(min(cfg.max_events, max(1, base)))


class _ChurnState:
    def __init__(self, ids: list[str], vectors: np.ndarray, payloads: dict[str, dict[str, Any]], next_id: int) -> None:
        self.ids = ids
        self.id_set = set(ids)
        self.vectors = vectors
        self.payloads = payloads
        self.next_id = next_id
        self.id_to_index = {doc_id: idx for idx, doc_id in enumerate(ids)}

    @classmethod
    def from_dataset(cls, dataset) -> "_ChurnState":  # type: ignore[no-untyped-def]
        ids = list(dataset.ids)
        vectors = np.asarray(dataset.vectors, dtype=np.float32)
        payloads = {ids[i]: dict(dataset.payloads[i]) for i in range(len(ids))}
        return cls(ids=ids, vectors=vectors, payloads=payloads, next_id=len(ids))

    def exact_topk(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        if not self.ids:
            return []
        scores = self.vectors @ query_vec
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [self.ids[int(i)] for i in order]

    def sample_id(self, rng: np.random.Generator) -> str | None:
        if not self.ids:
            return None
        return self.ids[int(rng.integers(0, len(self.ids)))]

    def insert_one(self, rng: np.random.Generator) -> None:
        new_id = f"doc-{self.next_id:07d}"
        self.next_id += 1
        vec = _random_unit_vector(self.vectors.shape[1], rng)
        payload = {
            "tenant_id": f"tenant-{int(rng.integers(0, 100)):03d}",
            "acl_bucket": int(rng.integers(0, 16)),
            "time_bucket": int(rng.integers(0, 52)),
        }
        self.ids.append(new_id)
        self.id_set.add(new_id)
        self.payloads[new_id] = payload
        self.vectors = np.vstack([self.vectors, vec[None, :]])
        self.id_to_index[new_id] = len(self.ids) - 1

    def latest_record(self) -> UpsertRecord:
        doc_id = self.ids[-1]
        idx = self.id_to_index[doc_id]
        return UpsertRecord(id=doc_id, vector=self.vectors[idx].tolist(), payload=self.payloads[doc_id])

    def update_vector(self, doc_id: str, vec: np.ndarray) -> None:
        if doc_id not in self.id_to_index:
            return
        self.vectors[self.id_to_index[doc_id]] = vec

    def delete_id(self, doc_id: str) -> None:
        if doc_id not in self.id_to_index:
            return
        idx = self.id_to_index[doc_id]
        last_idx = len(self.ids) - 1
        last_id = self.ids[last_idx]
        # Keep delete/update bookkeeping near O(1) by swapping with the tail.
        if idx != last_idx:
            self.ids[idx] = last_id
            self.vectors[idx] = self.vectors[last_idx]
            self.id_to_index[last_id] = idx
        self.ids.pop()
        self.vectors = self.vectors[:-1]
        self.payloads.pop(doc_id, None)
        self.id_set.discard(doc_id)
        self.id_to_index.pop(doc_id, None)


def _ingest_state(adapter: Any, state: _ChurnState) -> None:
    records = [
        UpsertRecord(id=doc_id, vector=state.vectors[i].tolist(), payload=state.payloads[doc_id])
        for i, doc_id in enumerate(state.ids)
    ]
    adapter.bulk_upsert(records)
    adapter.flush_or_commit()


def _sample_operation(cfg: S3Config, rng: np.random.Generator, *, write_multiplier: float) -> str:
    read = cfg.read_rate
    insert = cfg.insert_rate * write_multiplier
    update = cfg.update_rate * write_multiplier
    delete = cfg.delete_rate * write_multiplier
    total = max(read + insert + update + delete, 1e-9)
    draw = float(rng.random()) * total
    if draw < read:
        return "read"
    draw -= read
    if draw < insert:
        return "insert"
    draw -= insert
    if draw < update:
        return "update"
    return "delete"


def _random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(dim, dtype=np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def _ann_relevance_from_exact(*, exact_ids: list[str], k: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for rank, doc_id in enumerate(exact_ids[:k]):
        values[doc_id] = float(max(1, k - rank))
    return values
