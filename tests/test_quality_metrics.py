from __future__ import annotations

import pytest

from maxionbench.metrics.quality import recall_at_k


def test_recall_at_k_uses_full_ground_truth_denominator() -> None:
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"]

    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(5.0 / 12.0)


def test_recall_at_k_uses_only_top_k_retrieved_documents() -> None:
    retrieved = ["miss", "d1", "d2", "d3", "d4", "d5", "d6"]
    relevant = ["d1", "d2", "d3", "d4", "d5", "d6"]

    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 6.0)
