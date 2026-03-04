from __future__ import annotations

import importlib.util

import pytest

from maxionbench.adapters.qdrant import QdrantAdapter


def test_qdrant_metric_mapping() -> None:
    assert QdrantAdapter._distance_name("ip") == "Dot"
    assert QdrantAdapter._distance_name("l2") == "Euclid"
    assert QdrantAdapter._distance_name("cosine") == "Cosine"


def test_qdrant_filter_translation() -> None:
    filt = QdrantAdapter._to_filter({"tenant_id": "a", "acl": 3})
    assert "must" in filt
    assert len(filt["must"]) == 2


def test_qdrant_local_id_roundtrip_helpers() -> None:
    adapter = QdrantAdapter()
    encoded = adapter._encode_local_id("doc-1")
    decoded = adapter._decode_local_id(encoded, {"__maxionbench_id": "doc-1"})
    assert decoded == "doc-1"


def test_qdrant_local_mode_requires_qdrant_client_if_unavailable() -> None:
    if importlib.util.find_spec("qdrant_client") is not None:
        pytest.skip("qdrant-client installed in environment")
    with pytest.raises(ImportError):
        QdrantAdapter(location=":memory:")
