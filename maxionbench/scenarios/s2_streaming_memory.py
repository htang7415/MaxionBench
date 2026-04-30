"""Portable S2 streaming-memory scenario."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import numpy as np

from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset, tokenize_text, top_relevant_ids
from maxionbench.metrics.latency import latency_summary
from maxionbench.scenarios.portable_text_retrieval import PortableTextConfig, PortableTextResult, evaluate_text_queries, ingest_text_dataset
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord

_FRESHNESS_PROBE_DELAYS_S: tuple[float, float] = (1.0, 5.0)
_VISIBILITY_POLL_INTERVAL_S: float = 0.1


@dataclass(frozen=True)
class StreamingMemoryConfig:
    top_k: int
    clients_read: int
    clients_write: int
    sla_threshold_ms: float
    warmup_s: float = 0.0
    steady_state_s: float = 0.0
    phase_timing_mode: str = "bounded"
    phase_max_requests_per_phase: int | None = None
    max_freshness_events: int | None = None
    search_params: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StreamingMemoryResult:
    static: PortableTextResult
    freshness_hit_at_1s: float
    freshness_hit_at_5s: float
    stale_answer_rate_at_5s: float
    p95_visibility_latency_ms: float
    event_count: int
    overlap_skipped_event_count: int = 0

    def __post_init__(self) -> None:
        if math.isnan(self.freshness_hit_at_5s) and math.isnan(self.stale_answer_rate_at_5s):
            return
        if not np.isclose(self.stale_answer_rate_at_5s, 1.0 - self.freshness_hit_at_5s, atol=1e-9):
            raise ValueError("stale_answer_rate_at_5s must equal 1 - freshness_hit_at_5s")


def run(
    *,
    adapter: Any,
    cfg: StreamingMemoryConfig,
    background: D4RetrievalDataset,
    events: D4RetrievalDataset,
    static_observation_sink: Callable[[Mapping[str, Any]], None] | None = None,
    freshness_observation_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> StreamingMemoryResult:
    ingest_text_dataset(adapter, background)
    static = evaluate_text_queries(
        adapter=adapter,
        cfg=PortableTextConfig(
            top_k=cfg.top_k,
            clients_read=cfg.clients_read,
            sla_threshold_ms=cfg.sla_threshold_ms,
            warmup_s=cfg.warmup_s,
            steady_state_s=cfg.steady_state_s,
            phase_timing_mode=cfg.phase_timing_mode,
            phase_max_requests_per_phase=cfg.phase_max_requests_per_phase,
            search_params=cfg.search_params,
        ),
        dataset=background,
        observation_sink=static_observation_sink,
    )

    adapter.set_search_params(cfg.search_params or {})
    hit_1s: list[float] = []
    hit_5s: list[float] = []
    visibility_latencies_ms: list[float] = []
    background_doc_ids = {str(doc_id) for doc_id in background.doc_ids}
    overlap_skipped_event_count = 0
    event_count = min(len(events.query_ids), len(events.query_vectors))
    if cfg.max_freshness_events is not None:
        event_count = min(event_count, int(cfg.max_freshness_events))
    for event_idx in range(event_count):
        qid = events.query_ids[event_idx]
        qrels = events.qrels[qid]
        evidence_ids = top_relevant_ids(qrels, k=1)
        if not evidence_ids:
            continue
        evidence_id = evidence_ids[0]
        if evidence_id in background_doc_ids:
            overlap_skipped_event_count += 1
            continue
        try:
            doc_idx = events.doc_ids.index(evidence_id)
        except ValueError:
            continue
        adapter.insert(
            UpsertRecord(
                id=evidence_id,
                vector=events.doc_vectors[doc_idx].tolist(),
                payload={
                    "text": events.doc_texts[doc_idx],
                    "token_count": len(tokenize_text(events.doc_texts[doc_idx])),
                    "event_source": "streaming_memory",
                    "freshness_probe_query_id": str(qid),
                },
            )
        )
        adapter.flush_or_commit()
        query_vector = events.query_vectors[event_idx]
        visibility_s, probe_hit_1s, probe_hit_5s = _measure_freshness(
            adapter=adapter,
            query_vector=query_vector,
            top_k=cfg.top_k,
            target_doc_id=evidence_id,
        )
        visibility_latencies_ms.append(visibility_s * 1000.0)
        hit_1s.append(float(probe_hit_1s))
        hit_5s.append(float(probe_hit_5s))
        if freshness_observation_sink is not None:
            freshness_observation_sink(
                {
                    "observation_type": "freshness",
                    "event_index": int(event_idx),
                    "query_id": str(qid),
                    "target_doc_id": str(evidence_id),
                    "visibility_latency_ms": float(visibility_s * 1000.0),
                    "freshness_hit_at_1s": int(probe_hit_1s),
                    "freshness_hit_at_5s": int(probe_hit_5s),
                }
            )

    visibility_summary = latency_summary(visibility_latencies_ms)
    freshness_hit_at_1s = float(np.mean(np.asarray(hit_1s, dtype=np.float64))) if hit_1s else 0.0
    freshness_hit_at_5s = float(np.mean(np.asarray(hit_5s, dtype=np.float64))) if hit_5s else 0.0
    return StreamingMemoryResult(
        static=static,
        freshness_hit_at_1s=freshness_hit_at_1s,
        freshness_hit_at_5s=freshness_hit_at_5s,
        stale_answer_rate_at_5s=1.0 - freshness_hit_at_5s if hit_5s else 0.0,
        p95_visibility_latency_ms=visibility_summary["p95_ms"],
        event_count=len(hit_5s),
        overlap_skipped_event_count=overlap_skipped_event_count,
    )


def _measure_freshness(
    *,
    adapter: Any,
    query_vector: np.ndarray,
    top_k: int,
    target_doc_id: str,
) -> tuple[float, bool, bool]:
    max_delay_s = max(_FRESHNESS_PROBE_DELAYS_S)
    started = time.perf_counter()
    hit_1s = False
    hit_5s = False
    visibility_s: float | None = None
    scheduled_probe_s = 0.0
    while True:
        next_named_probe_s = min(
            (probe_s for probe_s in _FRESHNESS_PROBE_DELAYS_S if probe_s > scheduled_probe_s),
            default=max_delay_s,
        )
        next_probe_s = min(scheduled_probe_s, next_named_probe_s)
        remaining = next_probe_s - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        elapsed = time.perf_counter() - started
        found = _query_contains(adapter=adapter, query_vector=query_vector, top_k=top_k, target_doc_id=target_doc_id)
        if found and visibility_s is None:
            visibility_s = elapsed
        if not hit_1s and elapsed >= _FRESHNESS_PROBE_DELAYS_S[0] - 1e-6:
            hit_1s = found
        if elapsed >= _FRESHNESS_PROBE_DELAYS_S[1] - 1e-6:
            hit_5s = found
            break
        scheduled_probe_s = min(max_delay_s, elapsed + _VISIBILITY_POLL_INTERVAL_S)
    return (visibility_s if visibility_s is not None else max_delay_s), hit_1s, hit_5s


def _query_contains(
    *,
    adapter: Any,
    query_vector: np.ndarray,
    top_k: int,
    target_doc_id: str,
) -> bool:
    rows = adapter.query(QueryRequest(vector=query_vector.tolist(), top_k=max(top_k, 10)))
    return any(row.id == target_doc_id for row in rows[:10])
