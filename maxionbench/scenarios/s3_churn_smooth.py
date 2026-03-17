"""S3 dynamic churn smooth workload on D3-style metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from queue import Empty, Full, Queue
import threading
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
    clients_read: int = 1
    clients_write: int = 0
    max_events: int = 5000
    max_pending_ops: int | None = None


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


@dataclass(frozen=True)
class _ReadTask:
    scheduled_at: float
    phase: str
    vector: np.ndarray
    top_k: int
    expected_ids: list[str] | None
    relevance: dict[str, float] | None


@dataclass(frozen=True)
class _WriteTask:
    scheduled_at: float
    phase: str
    op: str
    record: UpsertRecord | None = None
    ids: list[str] | None = None
    vectors: list[list[float]] | None = None


@dataclass(frozen=True)
class _ReadOutcome:
    phase: str
    latency_ms: float
    error: bool
    got_ids: list[str]
    expected_ids: list[str] | None
    relevance: dict[str, float] | None


@dataclass(frozen=True)
class _WriteOutcome:
    phase: str
    error: bool


@dataclass(frozen=True)
class _PhaseSummary:
    total_events: int
    read_arrivals: int
    completed_reads: int
    errors: int
    overflows: int
    elapsed_s: float
    latencies_ms: list[float]
    recalls: list[float]
    ndcgs: list[float]
    mrrs: list[float]
    maintenance_calls: int
    maintenance_elapsed_s: float
    audit_sample_size: int


_STOP = object()


def run(
    adapter: Any,
    cfg: S3Config,
    rng: np.random.Generator,
    *,
    d3_params: D3Params,
    burst_multiplier_fn: Callable[[float], float] | None = None,
    vectors: np.ndarray | None = None,
    dataset: Any | None = None,
) -> S3Result:
    if dataset is None:
        if vectors is None:
            vectors = generate_synthetic_vectors(
                num_vectors=cfg.num_vectors,
                dim=cfg.vector_dim,
                seed=int(rng.integers(1, 1_000_000)),
            )
        dataset = generate_d3_dataset(vectors, d3_params)
    state = _OracleState.from_dataset(dataset)
    _ingest_state(adapter, state)

    write_workers = max(
        0,
        int(cfg.clients_write) if (cfg.insert_rate + cfg.update_rate + cfg.delete_rate) <= 0.0 else max(1, int(cfg.clients_write)),
    )
    queue_capacity = cfg.max_pending_ops or max(1, (cfg.clients_read + write_workers) * 10)
    read_queue: Queue[object] = Queue(maxsize=queue_capacity)
    write_queue: Queue[object] = Queue(maxsize=queue_capacity)
    read_outcomes: list[_ReadOutcome] = []
    write_outcomes: list[_WriteOutcome] = []
    read_outcomes_lock = threading.Lock()
    write_outcomes_lock = threading.Lock()

    read_threads = [
        threading.Thread(
            target=_read_worker,
            args=(adapter, read_queue, read_outcomes, read_outcomes_lock),
            daemon=True,
        )
        for _ in range(max(1, int(cfg.clients_read)))
    ]
    write_threads = [
        threading.Thread(
            target=_write_worker,
            args=(adapter, write_queue, write_outcomes, write_outcomes_lock),
            daemon=True,
        )
        for _ in range(write_workers)
    ]
    for thread in [*read_threads, *write_threads]:
        thread.start()

    try:
        warmup = _run_phase(
            adapter=adapter,
            cfg=cfg,
            rng=rng,
            state=state,
            phase="warmup",
            phase_s=float(cfg.warmup_s),
            collect_metrics=False,
            burst_multiplier_fn=burst_multiplier_fn,
            read_queue=read_queue,
            write_queue=write_queue,
            read_outcomes=read_outcomes,
            write_outcomes=write_outcomes,
        )
        measure = _run_phase(
            adapter=adapter,
            cfg=cfg,
            rng=rng,
            state=state,
            phase="measurement",
            phase_s=float(cfg.steady_state_s),
            collect_metrics=True,
            burst_multiplier_fn=burst_multiplier_fn,
            read_queue=read_queue,
            write_queue=write_queue,
            read_outcomes=read_outcomes,
            write_outcomes=write_outcomes,
        )
    finally:
        for _ in read_threads:
            read_queue.put(_STOP)
        for _ in write_threads:
            write_queue.put(_STOP)
        for thread in [*read_threads, *write_threads]:
            thread.join(timeout=5.0)

    elapsed = max(measure.elapsed_s, 1e-9)
    summary = latency_summary(measure.latencies_ms)
    over_sla = sum(1 for x in measure.latencies_ms if x > cfg.sla_threshold_ms)
    mean_recall = float(np.mean(np.asarray(measure.recalls, dtype=np.float64))) if measure.recalls else 0.0
    mean_ndcg = float(np.mean(np.asarray(measure.ndcgs, dtype=np.float64))) if measure.ndcgs else 0.0
    mean_mrr = float(np.mean(np.asarray(measure.mrrs, dtype=np.float64))) if measure.mrrs else 0.0
    info = {
        "mode": "s3_smooth",
        "events": measure.total_events,
        "reads": measure.read_arrivals,
        "queue_capacity": queue_capacity,
        "queue_overflow_errors": measure.overflows,
        "audit_sample_size": measure.audit_sample_size,
        "maintenance_calls": measure.maintenance_calls,
        "maintenance_elapsed_s": measure.maintenance_elapsed_s,
        "lambda_req_s": cfg.lambda_req_s,
        "read_rate": cfg.read_rate,
        "insert_rate": cfg.insert_rate,
        "update_rate": cfg.update_rate,
        "delete_rate": cfg.delete_rate,
        "burst_clock_anchor": "measurement_start",
        "phase": {
            "warmup_events": warmup.total_events,
            "warmup_elapsed_s": warmup.elapsed_s,
            "measure_events": measure.total_events,
            "measure_elapsed_s": measure.elapsed_s,
        },
    }
    return S3Result(
        p50_ms=summary["p50_ms"],
        p95_ms=summary["p95_ms"],
        p99_ms=summary["p99_ms"],
        qps=float(measure.completed_reads) / elapsed,
        recall_at_10=mean_recall,
        ndcg_at_10=mean_ndcg,
        mrr_at_10=mean_mrr,
        sla_violation_rate=sla_violation_rate(
            total_requests=max(1, measure.read_arrivals),
            over_sla=over_sla,
            errors=measure.errors,
        ),
        errors=measure.errors,
        info_json=json.dumps(info, sort_keys=True),
        measured_requests=measure.read_arrivals,
        measured_elapsed_s=measure.elapsed_s,
        warmup_requests=warmup.read_arrivals,
        warmup_elapsed_s=warmup.elapsed_s,
    )


def _run_phase(
    *,
    adapter: Any,
    cfg: S3Config,
    rng: np.random.Generator,
    state: "_OracleState",
    phase: str,
    phase_s: float,
    collect_metrics: bool,
    burst_multiplier_fn: Callable[[float], float] | None,
    read_queue: Queue[object],
    write_queue: Queue[object],
    read_outcomes: list[_ReadOutcome],
    write_outcomes: list[_WriteOutcome],
) -> _PhaseSummary:
    start_read_idx = len(read_outcomes)
    start_write_idx = len(write_outcomes)
    phase_start = time.perf_counter()
    phase_deadline = phase_start + max(0.0, phase_s)
    strict_timing = str(cfg.phase_timing_mode).strip().lower() == "strict"
    total_events = 0
    read_arrivals = 0
    overflows = 0
    next_scheduled_at = phase_start
    audit_every = _audit_interval(cfg=cfg, phase_s=phase_s) if collect_metrics else 0
    audited_reads = 0
    maintenance_state = _MaintenanceState()
    maintenance_stop = threading.Event()
    maintenance_thread: threading.Thread | None = None
    if collect_metrics and phase_s > 0.0:
        maintenance_thread = threading.Thread(
            target=_maintenance_loop,
            args=(adapter, cfg.maintenance_interval_s, phase_start, phase_deadline, maintenance_stop, maintenance_state),
            daemon=True,
        )
        maintenance_thread.start()

    try:
        while True:
            if not strict_timing and total_events >= int(cfg.max_events):
                break
            phase_clock_s = max(0.0, next_scheduled_at - phase_start)
            write_multiplier = burst_multiplier_fn(phase_clock_s) if burst_multiplier_fn else 1.0
            total_rate = _effective_total_rate(cfg, write_multiplier=write_multiplier)
            if total_rate <= 0.0:
                break
            next_scheduled_at += float(rng.exponential(1.0 / total_rate))
            if phase_s > 0.0 and next_scheduled_at > phase_deadline:
                break
            if phase_s <= 0.0 and total_events > 0:
                break
            _sleep_until(next_scheduled_at)
            total_events += 1
            op = _sample_operation(cfg, rng, write_multiplier=write_multiplier)
            if op == "read":
                query_vec = state.sample_query_vector(rng)
                if query_vec is None:
                    continue
                read_arrivals += 1
                expected_ids: list[str] | None = None
                relevance: dict[str, float] | None = None
                should_audit = collect_metrics and audit_every > 0 and ((read_arrivals - 1) % audit_every == 0)
                if should_audit:
                    expected_ids = state.exact_topk(query_vec, cfg.top_k)
                    relevance = _ann_relevance_from_exact(exact_ids=expected_ids, k=min(10, cfg.top_k))
                task = _ReadTask(
                    scheduled_at=next_scheduled_at,
                    phase=phase,
                    vector=query_vec,
                    top_k=cfg.top_k,
                    expected_ids=expected_ids,
                    relevance=relevance,
                )
                try:
                    read_queue.put_nowait(task)
                    if should_audit:
                        audited_reads += 1
                except Full:
                    overflows += 1
                    if should_audit:
                        audited_reads += 1
                continue

            task, apply_state = _prepare_write_task(state=state, cfg=cfg, rng=rng, op=op, scheduled_at=next_scheduled_at, phase=phase)
            if task is None or apply_state is None:
                continue
            try:
                write_queue.put_nowait(task)
                apply_state()
            except Full:
                overflows += 1
    finally:
        read_queue.join()
        write_queue.join()
        if maintenance_thread is not None:
            maintenance_stop.set()
            maintenance_thread.join(timeout=5.0)

    elapsed_s = max(0.0, time.perf_counter() - phase_start)
    phase_read_outcomes = read_outcomes[start_read_idx:]
    phase_write_outcomes = write_outcomes[start_write_idx:]
    latencies_ms = [item.latency_ms for item in phase_read_outcomes] if collect_metrics else []
    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    if collect_metrics:
        for item in phase_read_outcomes:
            if item.expected_ids is None or item.relevance is None:
                continue
            k_eval = min(10, cfg.top_k)
            recalls.append(recall_at_k(item.got_ids, item.expected_ids, k=k_eval))
            ndcgs.append(ndcg_at_10(item.got_ids, item.relevance))
            mrrs.append(mrr_at_k(item.got_ids, item.expected_ids, k=k_eval))
    errors = overflows + sum(1 for item in phase_read_outcomes if item.error) + sum(1 for item in phase_write_outcomes if item.error)
    return _PhaseSummary(
        total_events=total_events,
        read_arrivals=read_arrivals,
        completed_reads=len(phase_read_outcomes),
        errors=errors,
        overflows=overflows,
        elapsed_s=elapsed_s,
        latencies_ms=latencies_ms,
        recalls=recalls,
        ndcgs=ndcgs,
        mrrs=mrrs,
        maintenance_calls=maintenance_state.calls if collect_metrics else 0,
        maintenance_elapsed_s=maintenance_state.elapsed_s if collect_metrics else 0.0,
        audit_sample_size=audited_reads if collect_metrics else 0,
    )


def _sleep_until(target_ts: float) -> None:
    while True:
        remaining = target_ts - time.perf_counter()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.01))


class _MaintenanceState:
    def __init__(self) -> None:
        self.calls = 0
        self.elapsed_s = 0.0
        self._lock = threading.Lock()

    def record(self, duration_s: float) -> None:
        with self._lock:
            self.calls += 1
            self.elapsed_s += float(duration_s)


def _maintenance_loop(
    adapter: Any,
    interval_s: float,
    phase_start: float,
    phase_deadline: float,
    stop_event: threading.Event,
    state: _MaintenanceState,
) -> None:
    if interval_s <= 0.0:
        return
    next_trigger = phase_start + float(interval_s)
    while next_trigger < phase_deadline and not stop_event.is_set():
        wait_s = next_trigger - time.perf_counter()
        if wait_s > 0.0:
            if stop_event.wait(wait_s):
                return
        if stop_event.is_set():
            return
        started = time.perf_counter()
        try:
            adapter.optimize_or_compact()
        except Exception:
            pass
        finally:
            state.record(time.perf_counter() - started)
        next_trigger += float(interval_s)


def _read_worker(
    adapter: Any,
    task_queue: Queue[object],
    outcomes: list[_ReadOutcome],
    outcomes_lock: threading.Lock,
) -> None:
    while True:
        task = task_queue.get()
        try:
            if task is _STOP:
                return
            assert isinstance(task, _ReadTask)
            try:
                rows = adapter.query(QueryRequest(vector=task.vector.tolist(), top_k=task.top_k))
                got_ids = [row.id for row in rows]
                error = False
            except Exception:
                got_ids = []
                error = True
            latency_ms = (time.perf_counter() - task.scheduled_at) * 1000.0
            with outcomes_lock:
                outcomes.append(
                    _ReadOutcome(
                        phase=task.phase,
                        latency_ms=float(latency_ms),
                        error=error,
                        got_ids=got_ids,
                        expected_ids=task.expected_ids,
                        relevance=task.relevance,
                    )
                )
        finally:
            task_queue.task_done()


def _write_worker(
    adapter: Any,
    task_queue: Queue[object],
    outcomes: list[_WriteOutcome],
    outcomes_lock: threading.Lock,
) -> None:
    while True:
        task = task_queue.get()
        try:
            if task is _STOP:
                return
            assert isinstance(task, _WriteTask)
            error = False
            try:
                if task.op == "insert" and task.record is not None:
                    adapter.insert(task.record)
                    adapter.flush_or_commit()
                elif task.op == "update" and task.ids and task.vectors:
                    adapter.update_vectors(task.ids, task.vectors)
                    adapter.flush_or_commit()
                elif task.op == "delete" and task.ids:
                    adapter.delete(task.ids)
                    adapter.flush_or_commit()
            except Exception:
                error = True
            with outcomes_lock:
                outcomes.append(_WriteOutcome(phase=task.phase, error=error))
        finally:
            task_queue.task_done()


def _audit_interval(cfg: S3Config, *, phase_s: float) -> int:
    expected_reads = max(1, int(round(max(0.0, cfg.read_rate) * max(0.0, phase_s))))
    audit_budget = min(max(1, int(cfg.num_queries)), expected_reads)
    return max(1, int(np.ceil(expected_reads / max(1, audit_budget))))


def _effective_total_rate(cfg: S3Config, *, write_multiplier: float) -> float:
    return max(
        1e-9,
        float(cfg.read_rate)
        + (float(cfg.insert_rate) + float(cfg.update_rate) + float(cfg.delete_rate)) * float(write_multiplier),
    )


_OracleToken = int | str


class _OracleState:
    def __init__(self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[dict[str, Any]], next_id: int) -> None:
        self.base_ids = ids
        self.base_vectors = np.asarray(vectors, dtype=np.float32)
        self.base_payloads: Sequence[dict[str, Any]] | None = payloads
        self.base_active = np.ones(self.base_vectors.shape[0], dtype=bool)
        self.base_active_count = int(self.base_vectors.shape[0])
        self.updated_vectors: dict[int, np.ndarray] = {}
        self.inserted_ids: list[str] = []
        self.inserted_id_to_slot: dict[str, int] = {}
        self.inserted_vectors: dict[str, np.ndarray] = {}
        self.inserted_payloads: dict[str, dict[str, Any]] = {}
        self.next_id = next_id
        self.last_inserted_id: str | None = None

    @classmethod
    def from_dataset(cls, dataset) -> "_OracleState":  # type: ignore[no-untyped-def]
        vectors = np.asarray(dataset.vectors, dtype=np.float32)
        return cls(ids=dataset.ids, vectors=vectors, payloads=dataset.payloads, next_id=int(vectors.shape[0]))

    def sample_token(self, rng: np.random.Generator) -> _OracleToken | None:
        total_active = int(self.base_active_count) + len(self.inserted_ids)
        if total_active < 1:
            return None
        draw = int(rng.integers(0, total_active))
        if draw >= self.base_active_count:
            return self.inserted_ids[draw - self.base_active_count]
        if self.base_active_count >= self.base_vectors.shape[0]:
            return draw
        for _ in range(128):
            candidate = int(rng.integers(0, self.base_vectors.shape[0]))
            if self.base_active[candidate]:
                return candidate
        active_indices = np.flatnonzero(self.base_active)
        if active_indices.size == 0:
            return self.inserted_ids[0] if self.inserted_ids else None
        return int(active_indices[int(rng.integers(0, active_indices.size))])

    def sample_record_target(self, rng: np.random.Generator) -> tuple[_OracleToken, str] | None:
        token = self.sample_token(rng)
        if token is None:
            return None
        return token, self.doc_id_for_token(token)

    def sample_query_vector(self, rng: np.random.Generator) -> np.ndarray | None:
        token = self.sample_token(rng)
        if token is None:
            return None
        return self.vector_for_token(token).copy()

    def doc_id_for_token(self, token: _OracleToken) -> str:
        if isinstance(token, str):
            return token
        return self.base_ids[int(token)]

    def vector_for_token(self, token: _OracleToken) -> np.ndarray:
        if isinstance(token, str):
            return self.inserted_vectors[token]
        base_index = int(token)
        if base_index in self.updated_vectors:
            return self.updated_vectors[base_index]
        return self.base_vectors[base_index]

    def update_token(self, token: _OracleToken, vec: np.ndarray) -> None:
        if isinstance(token, str):
            if token not in self.inserted_id_to_slot:
                return
            self.inserted_vectors[token] = vec
            return
        base_index = int(token)
        if base_index < 0 or base_index >= self.base_vectors.shape[0] or not self.base_active[base_index]:
            return
        self.updated_vectors[base_index] = vec

    def delete_token(self, token: _OracleToken) -> None:
        if isinstance(token, str):
            if token not in self.inserted_id_to_slot:
                return
            slot = self.inserted_id_to_slot[token]
            last_id = self.inserted_ids[-1]
            if slot != len(self.inserted_ids) - 1:
                self.inserted_ids[slot] = last_id
                self.inserted_id_to_slot[last_id] = slot
            self.inserted_ids.pop()
            self.inserted_id_to_slot.pop(token, None)
            self.inserted_vectors.pop(token, None)
            self.inserted_payloads.pop(token, None)
            return
        base_index = int(token)
        if base_index < 0 or base_index >= self.base_vectors.shape[0] or not self.base_active[base_index]:
            return
        self.base_active[base_index] = False
        self.base_active_count -= 1
        self.updated_vectors.pop(base_index, None)

    def exact_topk(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        if (self.base_active_count + len(self.inserted_ids)) < 1 or top_k < 1:
            return []
        base_candidates: list[tuple[float, str]] = []
        if self.base_vectors.shape[0] > 0:
            base_scores = self.base_vectors @ query_vec
            invalid_count = int(self.base_vectors.shape[0] - self.base_active_count) + len(self.updated_vectors)
            candidate_k = min(self.base_vectors.shape[0], max(top_k, top_k + invalid_count))
            if candidate_k >= self.base_vectors.shape[0]:
                indices = np.arange(self.base_vectors.shape[0], dtype=np.int64)
            else:
                indices = np.argpartition(-base_scores, candidate_k - 1)[:candidate_k]
            ordered = sorted(
                (int(idx) for idx in indices.tolist()),
                key=lambda idx: (-float(base_scores[idx]), self.base_ids[idx]),
            )
            for idx in ordered:
                if not self.base_active[idx] or idx in self.updated_vectors:
                    continue
                base_candidates.append((float(base_scores[idx]), self.base_ids[idx]))
                if len(base_candidates) >= top_k:
                    break
        extras: list[tuple[float, str]] = []
        for idx, vec in self.updated_vectors.items():
            extras.append((float(np.dot(vec, query_vec)), self.base_ids[idx]))
        for doc_id, vec in self.inserted_vectors.items():
            extras.append((float(np.dot(vec, query_vec)), doc_id))
        merged = base_candidates + extras
        merged.sort(key=lambda item: (-item[0], item[1]))
        return [doc_id for _, doc_id in merged[:top_k]]

    def payload_for_base_index(self, index: int) -> dict[str, Any]:
        if self.base_payloads is None:
            raise RuntimeError("base payloads are no longer available")
        return dict(self.base_payloads[index])

    def release_base_payloads(self) -> None:
        self.base_payloads = None


def _prepare_write_task(
    *,
    state: _OracleState,
    cfg: S3Config,
    rng: np.random.Generator,
    op: str,
    scheduled_at: float,
    phase: str,
) -> tuple[_WriteTask | None, Callable[[], None] | None]:
    if op == "insert":
        inserted_id = f"doc-{state.next_id:07d}"
        inserted_vec = _random_unit_vector(cfg.vector_dim, rng)
        inserted_payload = {
            "tenant_id": f"tenant-{int(rng.integers(0, 100)):03d}",
            "acl_bucket": int(rng.integers(0, 16)),
            "time_bucket": int(rng.integers(0, 52)),
        }
        record = UpsertRecord(id=inserted_id, vector=inserted_vec.tolist(), payload=inserted_payload)

        def apply_insert() -> None:
            state.next_id += 1
            state.inserted_ids.append(inserted_id)
            state.inserted_id_to_slot[inserted_id] = len(state.inserted_ids) - 1
            state.inserted_vectors[inserted_id] = inserted_vec
            state.inserted_payloads[inserted_id] = inserted_payload
            state.last_inserted_id = inserted_id

        return _WriteTask(scheduled_at=scheduled_at, phase=phase, op="insert", record=record), apply_insert

    target = state.sample_record_target(rng)
    if target is None:
        return None, None
    target_token, target_id = target
    if op == "update":
        new_vec = _random_unit_vector(cfg.vector_dim, rng)

        def apply_update() -> None:
            state.update_token(target_token, new_vec)

        return (
            _WriteTask(
                scheduled_at=scheduled_at,
                phase=phase,
                op="update",
                ids=[target_id],
                vectors=[new_vec.tolist()],
            ),
            apply_update,
        )
    if op == "delete":

        def apply_delete() -> None:
            state.delete_token(target_token)

        return _WriteTask(scheduled_at=scheduled_at, phase=phase, op="delete", ids=[target_id]), apply_delete
    return None, None


def _ingest_state(adapter: Any, state: _OracleState) -> None:
    batch_size = 10_000
    total = int(state.base_vectors.shape[0])
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        records = [
            UpsertRecord(
                id=state.base_ids[i],
                vector=state.base_vectors[i].tolist(),
                payload=state.payload_for_base_index(i),
            )
            for i in range(start, stop)
        ]
        adapter.bulk_upsert(records)
    adapter.flush_or_commit()
    state.release_base_payloads()


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
