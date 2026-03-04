"""Deterministic synthetic D4 text retrieval bundle for local benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class D4RetrievalDataset:
    doc_ids: list[str]
    doc_vectors: np.ndarray
    doc_texts: list[str]
    doc_token_sets: list[set[str]]
    query_ids: list[str]
    query_vectors: np.ndarray
    query_texts: list[str]
    query_token_sets: list[set[str]]
    qrels: dict[str, dict[str, int]]
    idf: dict[str, float]


# Backward-compatible alias.
D4SyntheticDataset = D4RetrievalDataset


def generate_d4_synthetic_dataset(
    *,
    num_docs: int,
    num_queries: int,
    vector_dim: int,
    seed: int,
    num_topics: int = 64,
) -> D4RetrievalDataset:
    if num_docs < 1:
        raise ValueError("num_docs must be >= 1")
    if num_queries < 1:
        raise ValueError("num_queries must be >= 1")
    if vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    if num_topics < 2:
        raise ValueError("num_topics must be >= 2")

    rng = np.random.default_rng(seed)
    topic_centroids = _random_unit_vectors(rng, count=num_topics, dim=vector_dim)

    doc_topics = rng.integers(0, num_topics, size=num_docs)
    doc_ids = [f"doc-{idx:07d}" for idx in range(num_docs)]
    doc_vectors = _topic_vectors(rng, topic_centroids, doc_topics, noise_scale=0.18)
    doc_texts = [_doc_text(topic=int(topic), index=idx, rng=rng) for idx, topic in enumerate(doc_topics)]
    doc_token_sets = [set(tokenize_text(text)) for text in doc_texts]
    idf = compute_idf(doc_token_sets)

    query_topics = rng.integers(0, num_topics, size=num_queries)
    query_ids = [f"query-{idx:05d}" for idx in range(num_queries)]
    query_vectors = _topic_vectors(rng, topic_centroids, query_topics, noise_scale=0.12)
    query_texts = [_query_text(topic=int(topic), index=idx, rng=rng) for idx, topic in enumerate(query_topics)]
    query_token_sets = [set(tokenize_text(text)) for text in query_texts]

    qrels = _build_qrels(
        query_ids=query_ids,
        query_vectors=query_vectors,
        doc_ids=doc_ids,
        doc_vectors=doc_vectors,
        graded_cutoffs=(3, 10, 25),
    )
    return D4RetrievalDataset(
        doc_ids=doc_ids,
        doc_vectors=doc_vectors,
        doc_texts=doc_texts,
        doc_token_sets=doc_token_sets,
        query_ids=query_ids,
        query_vectors=query_vectors,
        query_texts=query_texts,
        query_token_sets=query_token_sets,
        qrels=qrels,
        idf=idf,
    )


def lexical_score(
    query_terms: set[str],
    doc_terms: set[str],
    *,
    idf: dict[str, float],
) -> float:
    score = 0.0
    for term in query_terms.intersection(doc_terms):
        score += idf.get(term, 0.0)
    return score


def top_relevant_ids(qrels: dict[str, int], k: int) -> list[str]:
    ordered = sorted(qrels.items(), key=lambda item: (-int(item[1]), item[0]))
    return [doc_id for doc_id, _ in ordered[:k]]


def _random_unit_vectors(rng: np.random.Generator, *, count: int, dim: int) -> np.ndarray:
    arr = rng.standard_normal((count, dim), dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def _topic_vectors(
    rng: np.random.Generator,
    centroids: np.ndarray,
    topics: np.ndarray,
    *,
    noise_scale: float,
) -> np.ndarray:
    vectors = centroids[topics].copy()
    vectors += rng.standard_normal(vectors.shape, dtype=np.float32) * noise_scale
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors.astype(np.float32, copy=False)


def _topic_terms(topic: int) -> tuple[str, str, str]:
    return (
        f"topic-{topic:03d}",
        f"signal-{(topic * 7) % 97:02d}",
        f"domain-{(topic * 11) % 89:02d}",
    )


def _doc_text(topic: int, index: int, rng: np.random.Generator) -> str:
    t0, t1, t2 = _topic_terms(topic)
    noise = f"noise-{int(rng.integers(0, 2048)):04d}"
    return f"{t0} {t1} {t2} passage-{index:07d} {noise}"


def _query_text(topic: int, index: int, rng: np.random.Generator) -> str:
    t0, t1, _ = _topic_terms(topic)
    intent = f"intent-{int(rng.integers(0, 64)):03d}"
    return f"{t0} {t1} query-{index:05d} {intent}"


def tokenize_text(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def compute_idf(doc_terms: list[set[str]]) -> dict[str, float]:
    n_docs = float(len(doc_terms))
    dfs: dict[str, int] = {}
    for terms in doc_terms:
        for token in terms:
            dfs[token] = dfs.get(token, 0) + 1
    return {term: math.log((1.0 + n_docs) / (1.0 + float(df))) + 1.0 for term, df in dfs.items()}


def _build_qrels(
    *,
    query_ids: list[str],
    query_vectors: np.ndarray,
    doc_ids: list[str],
    doc_vectors: np.ndarray,
    graded_cutoffs: tuple[int, int, int],
) -> dict[str, dict[str, int]]:
    top3, top10, top25 = graded_cutoffs
    if not (1 <= top3 < top10 < top25):
        raise ValueError("graded_cutoffs must satisfy 1 <= top3 < top10 < top25")

    sims = query_vectors @ doc_vectors.T
    qrels: dict[str, dict[str, int]] = {}
    for q_idx, qid in enumerate(query_ids):
        order = np.argsort(-sims[q_idx], kind="stable")[:top25]
        graded: dict[str, int] = {}
        for rank, doc_idx in enumerate(order, start=1):
            if rank <= top3:
                rel = 3
            elif rank <= top10:
                rel = 2
            else:
                rel = 1
            graded[doc_ids[int(doc_idx)]] = rel
        qrels[qid] = graded
    return qrels
