"""LanceDB in-process adapter."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from ._exact import StoredPoint, normalize_metric, topk_exact
from .base import BaseAdapter


class LanceDbInprocAdapter(BaseAdapter):
    """LanceDB in-process adapter with explicit flush semantics."""

    def __init__(self, uri: str = "artifacts/lancedb/inproc") -> None:
        try:
            import lancedb  # type: ignore
        except ImportError as exc:
            raise ImportError("lancedb is required for LanceDbInprocAdapter. Install with `pip install lancedb`.") from exc
        self._lancedb = lancedb
        self._uri = str(Path(uri).resolve())
        Path(self._uri).mkdir(parents=True, exist_ok=True)
        self._db = self._lancedb.connect(self._uri)
        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._records: dict[str, StoredPoint] = {}
        self._pending_upserts: dict[str, StoredPoint] = {}
        self._pending_deletes: set[str] = set()
        self._deleted_total = 0
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}
        self._table: Any | None = None

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._dimension = int(dimension)
        self._metric = normalize_metric(metric)
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._table = None
        self._drop_table_if_exists()
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        if collection != self._collection and collection:
            try:
                self._db.drop_table(collection)
            except Exception:
                pass
            return
        self._drop_table_if_exists()
        self._collection = ""
        self._dimension = 0
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._table = None

    def reset(self, collection: str) -> None:
        dimension = self._dimension or 1
        metric = self._metric
        self.drop(collection)
        self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        return bool(self._collection)

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        for record in records:
            key = str(record.id)
            self._pending_upserts[key] = StoredPoint(vector=self._to_vector(record.vector), payload=dict(record.payload))
            self._pending_deletes.discard(key)
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        query_vec = self._to_vector(request.vector)
        # Deterministic exact fallback across all metrics and filter shapes.
        return topk_exact(
            records=self._records,
            query=query_vec,
            top_k=request.top_k,
            metric=self._metric,
            filters=request.filters,
        )

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests]

    def insert(self, record: UpsertRecord) -> None:
        self.bulk_upsert([record])

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have same length")
        updated = 0
        for doc_id, vector in zip(ids, vectors):
            key = str(doc_id)
            base = self._pending_upserts.get(key) or self._records.get(key)
            if base is None:
                continue
            self._pending_upserts[key] = StoredPoint(vector=self._to_vector(vector), payload=dict(base.payload))
            self._pending_deletes.discard(key)
            updated += 1
        return updated

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        updated = 0
        for doc_id in ids:
            key = str(doc_id)
            base = self._pending_upserts.get(key) or self._records.get(key)
            if base is None:
                continue
            merged = dict(base.payload)
            merged.update(payload)
            self._pending_upserts[key] = StoredPoint(vector=base.vector.copy(), payload=merged)
            self._pending_deletes.discard(key)
            updated += 1
        return updated

    def delete(self, ids: Sequence[str]) -> int:
        for doc_id in ids:
            key = str(doc_id)
            self._pending_deletes.add(key)
            self._pending_upserts.pop(key, None)
        return len(ids)

    def flush_or_commit(self) -> None:
        for doc_id in sorted(self._pending_deletes):
            if doc_id in self._records:
                self._deleted_total += 1
            self._records.pop(doc_id, None)
        for doc_id in sorted(self._pending_upserts):
            self._records[doc_id] = self._pending_upserts[doc_id]
        self._pending_deletes.clear()
        self._pending_upserts.clear()
        self._sync_table()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        # LanceDB compaction/optimize APIs vary by version; flush ensures durable state here.
        self.flush_or_commit()

    def stats(self) -> AdapterStats:
        vector_count = len(self._records)
        disk_size = self._disk_usage_bytes()
        ram_size = vector_count * self._dimension * 4
        return AdapterStats(
            vector_count=vector_count,
            deleted_count=self._deleted_total,
            index_size_bytes=disk_size,
            ram_usage_bytes=ram_size,
            disk_usage_bytes=disk_size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _sync_table(self) -> None:
        self._drop_table_if_exists()
        if not self._records:
            self._table = None
            return
        rows = [
            {
                "id": doc_id,
                "vector": point.vector.tolist(),
                "payload": dict(point.payload),
            }
            for doc_id, point in sorted(self._records.items(), key=lambda item: item[0])
        ]
        self._table = self._db.create_table(self._collection, data=rows, mode="overwrite")

    def _drop_table_if_exists(self) -> None:
        if not self._collection:
            return
        try:
            self._db.drop_table(self._collection)
        except Exception:
            pass

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    def _disk_usage_bytes(self) -> int:
        root = Path(self._uri)
        if not root.exists():
            return 0
        return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))
