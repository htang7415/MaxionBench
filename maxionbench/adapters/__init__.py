"""Adapter registry and loader."""

from __future__ import annotations

from typing import Mapping

from maxionbench.schemas.adapter_contract import AdapterContract

from .faiss_cpu import FaissCpuAdapter
from .faiss_gpu import FaissGpuAdapter
from .lancedb_inproc import LanceDbInprocAdapter
from .lancedb_service import LanceDbServiceAdapter
from .milvus import MilvusAdapter
from .mock import MockAdapter
from .opensearch import OpenSearchAdapter
from .pgvector import PgVectorAdapter
from .qdrant import QdrantAdapter
from .weaviate import WeaviateAdapter

_ADAPTERS: Mapping[str, type[AdapterContract]] = {
    "mock": MockAdapter,
    "faiss_cpu": FaissCpuAdapter,
    "faiss-cpu": FaissCpuAdapter,
    "faiss_gpu": FaissGpuAdapter,
    "faiss-gpu": FaissGpuAdapter,
    "lancedb_inproc": LanceDbInprocAdapter,
    "lancedb-inproc": LanceDbInprocAdapter,
    "lancedb_service": LanceDbServiceAdapter,
    "lancedb-service": LanceDbServiceAdapter,
    "milvus": MilvusAdapter,
    "opensearch": OpenSearchAdapter,
    "pgvector": PgVectorAdapter,
    "qdrant": QdrantAdapter,
    "weaviate": WeaviateAdapter,
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
    "FaissGpuAdapter",
    "LanceDbInprocAdapter",
    "LanceDbServiceAdapter",
    "MilvusAdapter",
    "MockAdapter",
    "OpenSearchAdapter",
    "PgVectorAdapter",
    "QdrantAdapter",
    "WeaviateAdapter",
]
