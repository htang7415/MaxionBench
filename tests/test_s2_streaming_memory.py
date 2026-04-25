from __future__ import annotations

import math

import numpy as np
import pytest

from maxionbench.datasets.loaders.d4_synthetic import D4RetrievalDataset
from maxionbench.scenarios.portable_text_retrieval import PortableTextResult
from maxionbench.scenarios.s2_streaming_memory import StreamingMemoryConfig, StreamingMemoryResult, run
from maxionbench.schemas.adapter_contract import QueryRequest, QueryResult


def test_streaming_memory_result_enforces_freshness_staleness_invariant() -> None:
    static = PortableTextResult(
        recall_at_10=0.0,
        ndcg_at_10=0.0,
        mrr_at_10=0.0,
        evidence_coverage_at_5=0.0,
        evidence_coverage_at_10=0.0,
        evidence_coverage_at_20=0.0,
        avg_retrieved_input_tokens=0.0,
        p50_ms=0.0,
        p95_ms=0.0,
        p99_ms=0.0,
        qps=0.0,
        sla_violation_rate=0.0,
        errors=0,
        warmup_elapsed_s=0.0,
        warmup_requests=0,
        measured_elapsed_s=0.0,
        measured_requests=0,
    )

    with pytest.raises(ValueError, match="stale_answer_rate_at_5s must equal 1 - freshness_hit_at_5s"):
        StreamingMemoryResult(
            static=static,
            freshness_hit_at_1s=0.2,
            freshness_hit_at_5s=0.7,
            stale_answer_rate_at_5s=0.4,
            p95_visibility_latency_ms=1.0,
            event_count=1,
        )


def test_streaming_memory_result_allows_nan_freshness_pair() -> None:
    static = PortableTextResult(
        recall_at_10=0.0,
        ndcg_at_10=0.0,
        mrr_at_10=0.0,
        evidence_coverage_at_5=0.0,
        evidence_coverage_at_10=0.0,
        evidence_coverage_at_20=0.0,
        avg_retrieved_input_tokens=0.0,
        p50_ms=0.0,
        p95_ms=0.0,
        p99_ms=0.0,
        qps=0.0,
        sla_violation_rate=0.0,
        errors=0,
        warmup_elapsed_s=0.0,
        warmup_requests=0,
        measured_elapsed_s=0.0,
        measured_requests=0,
    )

    result = StreamingMemoryResult(
        static=static,
        freshness_hit_at_1s=math.nan,
        freshness_hit_at_5s=math.nan,
        stale_answer_rate_at_5s=math.nan,
        p95_visibility_latency_ms=1.0,
        event_count=0,
    )

    assert math.isnan(result.freshness_hit_at_5s)
    assert math.isnan(result.stale_answer_rate_at_5s)


def test_streaming_memory_skips_overlapping_background_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    static = PortableTextResult(
        recall_at_10=0.0,
        ndcg_at_10=0.0,
        mrr_at_10=0.0,
        evidence_coverage_at_5=0.0,
        evidence_coverage_at_10=0.0,
        evidence_coverage_at_20=0.0,
        avg_retrieved_input_tokens=0.0,
        p50_ms=0.0,
        p95_ms=0.0,
        p99_ms=0.0,
        qps=0.0,
        sla_violation_rate=0.0,
        errors=0,
        warmup_elapsed_s=0.0,
        warmup_requests=0,
        measured_elapsed_s=0.0,
        measured_requests=0,
    )

    class _Adapter:
        def __init__(self) -> None:
            self.inserted_ids: list[str] = []

        def bulk_upsert(self, records):  # type: ignore[no-untyped-def]
            return len(records)

        def set_search_params(self, params):  # type: ignore[no-untyped-def]
            return None

        def query(self, request: QueryRequest) -> list[QueryResult]:
            return []

        def insert(self, record) -> None:  # type: ignore[no-untyped-def]
            self.inserted_ids.append(str(record.id))

        def flush_or_commit(self) -> None:
            return None

    background = D4RetrievalDataset(
        doc_ids=["shared-doc"],
        doc_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        doc_texts=["background"],
        doc_token_sets=[{"background"}],
        query_ids=["bg-query"],
        query_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_texts=["background"],
        query_token_sets=[{"background"}],
        qrels={"bg-query": {"shared-doc": 1}},
        idf={"background": 1.0},
    )
    events = D4RetrievalDataset(
        doc_ids=["shared-doc", "fresh-doc"],
        doc_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        doc_texts=["shared", "fresh"],
        doc_token_sets=[{"shared"}, {"fresh"}],
        query_ids=["q-overlap", "q-fresh"],
        query_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_texts=["shared", "fresh"],
        query_token_sets=[{"shared"}, {"fresh"}],
        qrels={
            "q-overlap": {"shared-doc": 1},
            "q-fresh": {"fresh-doc": 1},
        },
        idf={"shared": 1.0, "fresh": 1.0},
    )

    monkeypatch.setattr("maxionbench.scenarios.s2_streaming_memory.evaluate_text_queries", lambda **kwargs: static)
    monkeypatch.setattr("maxionbench.scenarios.s2_streaming_memory.ingest_text_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr("maxionbench.scenarios.s2_streaming_memory._measure_freshness", lambda **kwargs: (0.5, True, True))

    adapter = _Adapter()
    result = run(
        adapter=adapter,
        cfg=StreamingMemoryConfig(top_k=10, clients_read=1, clients_write=1, sla_threshold_ms=10.0),
        background=background,
        events=events,
    )

    assert adapter.inserted_ids == ["fresh-doc"]
    assert result.event_count == 1
    assert result.overlap_skipped_event_count == 1
