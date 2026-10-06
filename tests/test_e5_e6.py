from __future__ import annotations

from pathlib import Path

import pytest

from maxionbench.eval.e5 import ItemResult, shard_metrics
from maxionbench.eval.e6 import Answer, Session, _find_inlined, build_sessions, trial_metrics

HOTPOT = Path("dataset/processed/hotpot_portable")


def _rag(correct: bool, label: str, cost: float, em: float = 0.0) -> ItemResult:
    return ItemResult("crag", "x", "ok", correct, 0.5, 0.2, cost, {"em": em, "f1": em, "judge_label": label})


def test_shard_metrics_rag_cost_per_correct_and_crag_score() -> None:
    items = [_rag(True, "correct", 0.002, 1.0), _rag(False, "missing", 0.001), _rag(False, "incorrect", 0.001)]
    m = shard_metrics(items)
    assert m["accuracy"] == pytest.approx(1 / 3) and m["usd_per_correct"] == pytest.approx(0.004)
    assert m["crag_score"] == pytest.approx(0.0) and m["missing_rate"] == pytest.approx(1 / 3)
    assert m["ttft_p50_ms"] == pytest.approx(200.0) and m["em"] == pytest.approx(1 / 3)
    assert "usd_per_correct" not in shard_metrics([_rag(False, "missing", 0.001)])


def test_find_inlined_handles_nesting() -> None:
    items = [{"response": {}, "metadata": {"key": "a"}}]
    assert _find_inlined({"done": True, "response": {"inlinedResponses": {"inlinedResponses": items}}}) == items
    assert _find_inlined({"metadata": {"output": {"inlinedResponses": items}}}) == items
    assert _find_inlined({"done": True}) == []


def test_trial_metrics_cached_ratio_and_em() -> None:
    session = Session("s", "doc", (("q1", "Q?", "Paris"), ("q2", "Q2?", "Rome")))
    answers = [Answer("q1", "Paris", "ok", 0.3, 0.5, 5000, 0, 3), Answer("q2", "Milan", "ok", 0.1, 0.2, 5000, 4000, 3)]
    m = trial_metrics("implicit", answers, [session], spend=0.01, turnaround_s=0.0, storage_usd=0.0)
    assert m["cached_token_ratio"] == pytest.approx(0.4) and m["em"] == 0.5 and m["usd_per_1k_requests"] == 5.0
    assert "turnaround_s" not in m and m["ttft_p50_ms"] == pytest.approx(200.0)


@pytest.mark.skipif(not HOTPOT.exists(), reason="processed HotpotQA missing")
def test_sessions_share_a_document_with_every_gold_paragraph() -> None:
    a = build_sessions(2, 3, 60, seed=1)
    assert a == build_sessions(2, 3, 60, seed=1) and a != build_sessions(2, 3, 60, seed=2)
    assert all(len(s.questions) == 3 for s in a) and a[0].document != a[1].document
    assert len(a[0].document) // 4 > 4096  # over Gemini 3.x's caching minimum
