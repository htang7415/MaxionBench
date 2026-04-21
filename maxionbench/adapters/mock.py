"""Deterministic in-memory adapter for conformance and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
import time
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from .base import BaseAdapter


@dataclass
class _StoredRecord:
    vector: np.ndarray
    payload: dict[str, Any]


class MockAdapter(BaseAdapter):
    """In-memory adapter with explicit flush semantics.

    Writes are not visible until flush_or_commit() is called.
    """

    def __init__(self) -> None:
        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._lock = threading.RLock()
        self._records: dict[str, _StoredRecord] = {}
        self._pending_upserts: dict[str, _StoredRecord] = {}
        self._pending_deletes: set[str] = set()
        self._deleted_total = 0
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        with self._lock:
            self._collection = collection
            self._dimension = dimension
            self._metric = metric
            self._records.clear()
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._deleted_total = 0
            self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        with self._lock:
            if collection != self._collection:
                return
            self._collection = ""
            self._records.clear()
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._deleted_total = 0

    def reset(self, collection: str) -> None:
        with self._lock:
            self.drop(collection)
            self.create(collection=collection, dimension=self._dimension or 1, metric=self._metric)

    def healthcheck(self) -> bool:
        with self._lock:
            return bool(self._collection)

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        with self._lock:
            for record in records:
                self._pending_upserts[record.id] = self._to_stored_record(record.vector, record.payload)
                self._pending_deletes.discard(record.id)
            return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        with self._lock:
            query_vec = self._to_vector(request.vector)
            candidates: list[QueryResult] = []
            for doc_id, record in self._records.items():
                if not self._matches_filter(record.payload, request.filters):
                    continue
                score = self._score(query_vec, record.vector)
                candidates.append(QueryResult(id=doc_id, score=score, payload=dict(record.payload)))
            candidates.sort(key=lambda item: (-item.score, item.id))
            return candidates[: request.top_k]

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests]

    def insert(self, record: UpsertRecord) -> None:
        with self._lock:
            self._pending_upserts[record.id] = self._to_stored_record(record.vector, record.payload)
            self._pending_deletes.discard(record.id)

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        with self._lock:
            if len(ids) != len(vectors):
                raise ValueError("ids and vectors must have same length")
            updated = 0
            for doc_id, vector in zip(ids, vectors):
                base = self._pending_upserts.get(doc_id) or self._records.get(doc_id)
                if base is None:
                    continue
                self._pending_upserts[doc_id] = _StoredRecord(
                    vector=self._to_vector(vector),
                    payload=dict(base.payload),
                )
                self._pending_deletes.discard(doc_id)
                updated += 1
            return updated

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        with self._lock:
            updated = 0
            for doc_id in ids:
                base = self._pending_upserts.get(doc_id) or self._records.get(doc_id)
                if base is None:
                    continue
                merged_payload = dict(base.payload)
                merged_payload.update(payload)
                self._pending_upserts[doc_id] = _StoredRecord(vector=base.vector.copy(), payload=merged_payload)
                self._pending_deletes.discard(doc_id)
                updated += 1
            return updated

    def delete(self, ids: Sequence[str]) -> int:
        with self._lock:
            for doc_id in ids:
                self._pending_deletes.add(doc_id)
                self._pending_upserts.pop(doc_id, None)
            return len(ids)

    def flush_or_commit(self) -> None:
        with self._lock:
            for doc_id in sorted(self._pending_deletes):
                if doc_id in self._records:
                    self._deleted_total += 1
                self._records.pop(doc_id, None)
            for doc_id in sorted(self._pending_upserts):
                self._records[doc_id] = self._pending_upserts[doc_id]
            self._pending_deletes.clear()
            self._pending_upserts.clear()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        with self._lock:
            self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        with self._lock:
            self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        # No-op for in-memory adapter; included to verify interface behavior.
        return

    def stats(self) -> AdapterStats:
        with self._lock:
            vector_bytes = self._dimension * 4
            payload_bytes = 64
            index_size = len(self._records) * (vector_bytes + payload_bytes)
            return AdapterStats(
                vector_count=len(self._records),
                deleted_count=self._deleted_total,
                index_size_bytes=index_size,
                ram_usage_bytes=index_size,
                disk_usage_bytes=index_size,
                engine_uptime_s=time.monotonic() - self._created_at,
            )

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    def _score(self, query: np.ndarray, candidate: np.ndarray) -> float:
        if self._metric == "l2":
            return float(-np.linalg.norm(query - candidate))
        if self._metric in {"cos", "cosine"}:
            query_norm = float(np.linalg.norm(query)) + 1e-12
            candidate_norm = float(np.linalg.norm(candidate)) + 1e-12
            return float(np.dot(query, candidate) / (query_norm * candidate_norm))
        return float(np.dot(query, candidate))

    def _to_stored_record(self, vector: Vector, payload: Mapping[str, Any]) -> _StoredRecord:
        return _StoredRecord(vector=self._to_vector(vector), payload=dict(payload))

    @staticmethod
    def _matches_filter(payload: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if payload.get(key) != expected:
                return False
        return True
