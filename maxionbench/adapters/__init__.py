"""Adapter registry and loader."""

from __future__ import annotations

from typing import Mapping

from maxionbench.schemas.adapter_contract import AdapterContract

from .faiss_cpu import FaissCpuAdapter
from .lancedb_inproc import LanceDbInprocAdapter
from .lancedb_service import LanceDbServiceAdapter
from .mock import MockAdapter
from .pgvector import PgVectorAdapter
from .qdrant import QdrantAdapter

_ADAPTERS: Mapping[str, type[AdapterContract]] = {
    "mock": MockAdapter,
    "faiss_cpu": FaissCpuAdapter,
    "faiss-cpu": FaissCpuAdapter,
    "lancedb_inproc": LanceDbInprocAdapter,
    "lancedb-inproc": LanceDbInprocAdapter,
    "lancedb_service": LanceDbServiceAdapter,
    "lancedb-service": LanceDbServiceAdapter,
    "pgvector": PgVectorAdapter,
    "qdrant": QdrantAdapter,
}


def create_adapter(name: str, **kwargs: object) -> AdapterContract:
    key = name.strip().lower()
    if key not in _ADAPTERS:
        supported = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unsupported adapter '{name}'. Supported: {supported}")
    return _ADAPTERS[key](**kwargs)  # type: ignore[call-arg]


__all__ = [
    "create_adapter",
    "FaissCpuAdapter",
    "LanceDbInprocAdapter",
    "LanceDbServiceAdapter",
    "MockAdapter",
    "PgVectorAdapter",
    "QdrantAdapter",
]
