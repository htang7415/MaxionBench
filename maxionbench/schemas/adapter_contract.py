"""Adapter contract for MaxionBench engines.

This module is the single typed source of truth for adapter methods required by
project.md + prompt.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

Vector = Sequence[float]


@dataclass(frozen=True)
class UpsertRecord:
    """Record used by insert and bulk_upsert."""

    id: str
    vector: Vector
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRequest:
    """Request payload for retrieval queries."""

    vector: Vector
    top_k: int = 10
    filters: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QueryResult:
    """One ranked query result."""

    id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterStats:
    """Minimum stats fields required by prompt.md Section 5."""

    vector_count: int
    deleted_count: int
    index_size_bytes: int
    ram_usage_bytes: int
    disk_usage_bytes: int
    engine_uptime_s: float


class AdapterContract(Protocol):
    """Engine adapter API contract required by MaxionBench."""

    # Lifecycle
    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        ...

    def drop(self, collection: str) -> None:
        ...

    def reset(self, collection: str) -> None:
        ...

    def healthcheck(self) -> bool:
        ...

    # Data ops
    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        ...

    def query(self, request: QueryRequest) -> list[QueryResult]:
        ...

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        ...

    def insert(self, record: UpsertRecord) -> None:
        ...

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        ...

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        ...

    def delete(self, ids: Sequence[str]) -> int:
        ...

    def flush_or_commit(self) -> None:
        ...

    # Index/control
    def set_index_params(self, params: Mapping[str, Any]) -> None:
        ...

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        ...

    def optimize_or_compact(self) -> None:
        ...

    # Stats
    def stats(self) -> AdapterStats:
        ...
