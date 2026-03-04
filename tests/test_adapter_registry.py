from __future__ import annotations

import pytest

import maxionbench.adapters as adapters_mod
from maxionbench.adapters import create_adapter


def test_adapter_registry_contains_all_required_engines() -> None:
    expected = {
        "mock",
        "qdrant",
        "pgvector",
        "milvus",
        "weaviate",
        "opensearch",
        "lancedb-service",
        "lancedb-inproc",
        "faiss-cpu",
        "faiss-gpu",
    }
    assert expected.issubset(set(adapters_mod._ADAPTERS.keys()))


def test_create_adapter_unknown_name_fails_with_supported_list() -> None:
    with pytest.raises(ValueError) as exc:
        create_adapter("unknown-adapter")
    message = str(exc.value)
    assert "Supported:" in message
    assert "qdrant" in message
    assert "pgvector" in message


def test_service_style_adapters_construct_without_remote_connections() -> None:
    weaviate = create_adapter("weaviate")
    opensearch = create_adapter("opensearch")
    lancedb_service = create_adapter("lancedb-service", base_url="http://127.0.0.1:18080")

    assert weaviate is not None
    assert opensearch is not None
    assert lancedb_service is not None
