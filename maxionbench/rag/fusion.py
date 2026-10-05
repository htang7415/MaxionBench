"""Rank fusion for hybrid retrieval."""

from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, top_k: int, k: int = 60) -> list[str]:
    """Fuse ranked id lists with RRF; ties break by first appearance order."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores, key=lambda d: (-scores[d], order[d]))[:top_k]
