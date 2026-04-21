from __future__ import annotations

import pytest

import maxionbench.adapters as adapters_mod
from maxionbench.adapters import create_adapter


def test_adapter_registry_contains_portable_engines() -> None:
    expected = {
        "mock",
        "qdrant",
        "pgvector",
        "lancedb-service",
        "lancedb-inproc",
        "faiss-cpu",
        "lancedb_service",
        "lancedb_inproc",
        "faiss_cpu",
    }
    assert set(adapters_mod._ADAPTERS.keys()) == expected


def test_create_adapter_unknown_name_fails_with_supported_list() -> None:
    with pytest.raises(ValueError) as exc:
        create_adapter("unknown-adapter")
    message = str(exc.value)
    assert "Supported:" in message
    assert "qdrant" in message
    assert "pgvector" in message


def test_portable_service_style_adapter_constructs_without_remote_connection() -> None:
    lancedb_service = create_adapter("lancedb-service", base_url="http://127.0.0.1:18080")
    assert lancedb_service is not None
