"""Quality metrics used by MaxionBench."""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    gt = set(ground_truth_ids)
    if not gt:
        return 0.0
    hits = len(set(retrieved_ids[:k]).intersection(gt))
    return hits / len(gt)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_10(retrieved_ids: Sequence[str], relevance: dict[str, float]) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(retrieved_ids[:10], start=1):
        rel = relevance.get(doc_id, 0.0)
        dcg += (2.0**rel - 1.0) / math.log2(index + 1.0)
    ideal_rels = sorted(relevance.values(), reverse=True)[:10]
    if not ideal_rels:
        return 0.0
    idcg = 0.0
    for index, rel in enumerate(ideal_rels, start=1):
        idcg += (2.0**rel - 1.0) / math.log2(index + 1.0)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evidence_coverage_at_k(
    retrieved_ids: Sequence[str],
    evidence_ids: Sequence[str],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    evidence = set(evidence_ids)
    if not evidence:
        return 0.0
    hits = len(evidence.intersection(retrieved_ids[:k]))
    return hits / len(evidence)
