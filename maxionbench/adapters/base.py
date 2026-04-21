"""Abstract base class for adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)


class BaseAdapter(ABC):
    """Abstract adapter matching the required adapter contract."""

    @abstractmethod
    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        ...

    @abstractmethod
    def drop(self, collection: str) -> None:
        ...

    @abstractmethod
    def reset(self, collection: str) -> None:
        ...

    @abstractmethod
    def healthcheck(self) -> bool:
        ...

    @abstractmethod
    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        ...

    @abstractmethod
    def query(self, request: QueryRequest) -> list[QueryResult]:
        ...

    @abstractmethod
    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        ...

    @abstractmethod
    def insert(self, record: UpsertRecord) -> None:
        ...

    @abstractmethod
    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        ...

    @abstractmethod
    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        ...

    @abstractmethod
    def flush_or_commit(self) -> None:
        ...

    @abstractmethod
    def set_index_params(self, params: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def set_search_params(self, params: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def optimize_or_compact(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> AdapterStats:
        ...
