from __future__ import annotations

import math

import pytest

from maxionbench.scenarios.portable_text_retrieval import PortableTextResult
from maxionbench.scenarios.s2_streaming_memory import StreamingMemoryResult


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
