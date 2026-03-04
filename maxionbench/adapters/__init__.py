"""Adapter registry and loader."""

from __future__ import annotations

from typing import Mapping

from maxionbench.schemas.adapter_contract import AdapterContract

from .mock import MockAdapter
from .pgvector import PgVectorAdapter
from .qdrant import QdrantAdapter

_ADAPTERS: Mapping[str, type[AdapterContract]] = {
    "mock": MockAdapter,
    "pgvector": PgVectorAdapter,
    "qdrant": QdrantAdapter,
}


def create_adapter(name: str, **kwargs: object) -> AdapterContract:
    key = name.strip().lower()
    if key not in _ADAPTERS:
        supported = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unsupported adapter '{name}'. Supported: {supported}")
    return _ADAPTERS[key](**kwargs)  # type: ignore[call-arg]


__all__ = ["create_adapter", "MockAdapter", "PgVectorAdapter", "QdrantAdapter"]
