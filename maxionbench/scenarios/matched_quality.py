"""Matched-quality candidate selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class MatchedQualityCandidate:
    quality: float
    p99_ms: float
    qps: float
    rhu_h: float
    payload: Any


def select_candidate(
    candidates: Sequence[MatchedQualityCandidate],
    target_quality: float,
) -> MatchedQualityCandidate | None:
    feasible = [candidate for candidate in candidates if candidate.quality >= target_quality]
    if not feasible:
        return None
    feasible.sort(key=lambda item: (item.rhu_h, item.p99_ms, -item.qps))
    return feasible[0]
