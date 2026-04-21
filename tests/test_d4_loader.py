from __future__ import annotations

import numpy as np

from maxionbench.datasets.loaders.d4_synthetic import (
    generate_d4_synthetic_dataset,
    lexical_score,
    top_relevant_ids,
)


def test_d4_synthetic_loader_is_deterministic() -> None:
    a = generate_d4_synthetic_dataset(num_docs=300, num_queries=20, vector_dim=16, seed=17)
    b = generate_d4_synthetic_dataset(num_docs=300, num_queries=20, vector_dim=16, seed=17)

    assert a.doc_ids == b.doc_ids
    assert a.query_ids == b.query_ids
    assert a.doc_texts == b.doc_texts
    assert a.query_texts == b.query_texts
    np.testing.assert_allclose(a.doc_vectors, b.doc_vectors)
    np.testing.assert_allclose(a.query_vectors, b.query_vectors)
    assert a.qrels == b.qrels


def test_d4_qrels_and_lexical_helpers() -> None:
    dataset = generate_d4_synthetic_dataset(num_docs=200, num_queries=12, vector_dim=12, seed=29)
    qid = dataset.query_ids[0]
    qrels = dataset.qrels[qid]
    top = top_relevant_ids(qrels, k=10)
    assert len(top) == 10
    assert qrels[top[0]] >= qrels[top[-1]]

    terms = dataset.query_token_sets[0]
    score = lexical_score(terms, dataset.doc_token_sets[0], idf=dataset.idf)
    assert score >= 0.0
