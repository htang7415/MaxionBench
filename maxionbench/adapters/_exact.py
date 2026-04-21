"""Shared exact-scoring utilities for adapter fallback paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from maxionbench.schemas.adapter_contract import QueryResult


@dataclass
class StoredPoint:
    vector: np.ndarray
    payload: dict[str, object]


def normalize_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized in {"ip", "inner_product", "dot"}:
        return "ip"
    if normalized in {"l2", "euclid", "euclidean"}:
        return "l2"
    if normalized in {"cos", "cosine"}:
        return "cos"
    raise ValueError(f"Unsupported metric: {metric}")


def matches_filter(payload: Mapping[str, object], filters: Mapping[str, object] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if payload.get(key) != expected:
            return False
    return True


def topk_exact(
    *,
    records: Mapping[str, StoredPoint],
    query: np.ndarray,
    top_k: int,
    metric: str,
    filters: Mapping[str, object] | None,
) -> list[QueryResult]:
    scored: list[tuple[float, str, dict[str, object]]] = []
    normalized_metric = normalize_metric(metric)
    query_vec = np.asarray(query, dtype=np.float32)
    if normalized_metric == "cos":
        query_vec = _unit(query_vec)

    for doc_id in sorted(records.keys()):
        point = records[doc_id]
        if not matches_filter(point.payload, filters):
            continue
        cand = point.vector
        if normalized_metric == "cos":
            cand = _unit(cand)
        if normalized_metric == "l2":
            score = float(-np.linalg.norm(query_vec - cand))
        else:
            score = float(np.dot(query_vec, cand))
        scored.append((score, doc_id, dict(point.payload)))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [QueryResult(id=doc_id, score=score, payload=payload) for score, doc_id, payload in scored[:top_k]]


def _unit(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec)) + 1e-12
    return vec / denom
