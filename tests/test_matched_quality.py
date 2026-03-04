from __future__ import annotations

from maxionbench.scenarios.matched_quality import MatchedQualityCandidate, select_candidate


def test_select_candidate_prefers_lower_rhu_then_p99_then_higher_qps() -> None:
    candidates = [
        MatchedQualityCandidate(quality=0.92, p99_ms=12.0, qps=100.0, rhu_h=2.0, payload="a"),
        MatchedQualityCandidate(quality=0.93, p99_ms=11.0, qps=90.0, rhu_h=1.5, payload="b"),
        MatchedQualityCandidate(quality=0.91, p99_ms=9.0, qps=110.0, rhu_h=1.5, payload="c"),
    ]
    selected = select_candidate(candidates, target_quality=0.9)
    assert selected is not None
    assert selected.payload == "c"


def test_select_candidate_returns_none_when_no_feasible_option() -> None:
    candidates = [
        MatchedQualityCandidate(quality=0.70, p99_ms=10.0, qps=100.0, rhu_h=1.0, payload=None),
    ]
    assert select_candidate(candidates, target_quality=0.8) is None
