"""FAISS CPU adapter."""

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

from ._exact import StoredPoint, normalize_metric, topk_exact
from .base import BaseAdapter


@dataclass(frozen=True)
class _FaissCpuConfig:
    metric: str


class FaissCpuAdapter(BaseAdapter):
    """FAISS CPU adapter with exact filtered-query fallback."""

    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss is required for FaissCpuAdapter. Install with `pip install faiss-cpu`.") from exc
        self._faiss = faiss
        self._collection = ""
        self._dimension = 0
        self._cfg = _FaissCpuConfig(metric="ip")
        self._created_at = time.monotonic()
        self._lock = threading.RLock()
        self._records: dict[str, StoredPoint] = {}
        self._pending_upserts: dict[str, StoredPoint] = {}
        self._pending_deletes: set[str] = set()
        self._deleted_total = 0
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}
        self._index: Any | None = None
        self._id_by_pos: list[str] = []

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        with self._lock:
            self._collection = collection
            self._dimension = int(dimension)
            self._cfg = _FaissCpuConfig(metric=normalize_metric(metric))
            self._records.clear()
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._id_by_pos.clear()
            self._deleted_total = 0
            self._index = None
            self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        with self._lock:
            if collection != self._collection:
                return
            self._collection = ""
            self._dimension = 0
            self._records.clear()
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._id_by_pos.clear()
            self._index = None
            self._deleted_total = 0

    def reset(self, collection: str) -> None:
        with self._lock:
            dimension = self._dimension or 1
            metric = self._cfg.metric
            self.drop(collection)
            self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        with self._lock:
            return bool(self._collection)

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        with self._lock:
            for record in records:
                point = StoredPoint(vector=self._to_vector(record.vector), payload=dict(record.payload))
                self._pending_upserts[str(record.id)] = point
                self._pending_deletes.discard(str(record.id))
            return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        with self._lock:
            query_vec = self._to_vector(request.vector)
            if request.filters:
                return topk_exact(
                    records=self._records,
                    query=query_vec,
                    top_k=request.top_k,
                    metric=self._cfg.metric,
                    filters=request.filters,
                )
            if self._index is None:
                return []
            k = max(1, int(request.top_k))
            q = query_vec[None, :]
            if self._cfg.metric == "cos":
                q = self._normalize_rows(q)
            self._apply_search_params(self._index)
            distances, indices = self._index.search(q.astype(np.float32), k)
            results: list[QueryResult] = []
            for dist, pos in zip(distances[0], indices[0]):
                pos_int = int(pos)
                if pos_int < 0 or pos_int >= len(self._id_by_pos):
                    continue
                doc_id = self._id_by_pos[pos_int]
                payload = dict(self._records[doc_id].payload)
                score = float(-dist if self._cfg.metric == "l2" else dist)
                results.append(QueryResult(id=doc_id, score=score, payload=payload))
            return results

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests]

    def insert(self, record: UpsertRecord) -> None:
        with self._lock:
            self.bulk_upsert([record])

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            for doc_id in ids:
                key = str(doc_id)
                self._pending_deletes.add(key)
                self._pending_upserts.pop(key, None)
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
            self._rebuild_index()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        with self._lock:
            self._index_params = dict(params)
            self._rebuild_index()

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        with self._lock:
            self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        with self._lock:
            self._rebuild_index()

    def stats(self) -> AdapterStats:
        with self._lock:
            vector_count = len(self._records)
            vector_bytes = vector_count * self._dimension * 4
            payload_bytes = vector_count * 64
            size = vector_bytes + payload_bytes
            return AdapterStats(
                vector_count=vector_count,
                deleted_count=self._deleted_total,
                index_size_bytes=size,
                ram_usage_bytes=size,
                disk_usage_bytes=size,
                engine_uptime_s=time.monotonic() - self._created_at,
            )

    def _rebuild_index(self) -> None:
        if not self._records:
            self._index = None
            self._id_by_pos = []
            return
        ids = sorted(self._records.keys())
        vectors = np.vstack([self._records[doc_id].vector for doc_id in ids]).astype(np.float32)
        if self._cfg.metric == "cos":
            vectors = self._normalize_rows(vectors)
        cpu_index = self._build_cpu_index(vectors=vectors)
        self._index = self._finalize_index(cpu_index)
        self._id_by_pos = ids

    def _build_cpu_index(self, *, vectors: np.ndarray) -> Any:
        metric = self._faiss.METRIC_INNER_PRODUCT if self._cfg.metric in {"ip", "cos"} else self._faiss.METRIC_L2
        index_type = str(self._index_params.get("index_type", "flat")).strip().lower()

        if index_type == "hnsw":
            m = int(self._index_params.get("hnsw_m", self._index_params.get("M", 16)))
            index = self._faiss.IndexHNSWFlat(self._dimension, m, metric)
            ef_construction = self._index_params.get("hnsw_ef_construction", self._index_params.get("efConstruction"))
            if ef_construction is not None:
                index.hnsw.efConstruction = int(ef_construction)
        elif index_type == "ivf":
            nlist = int(self._index_params.get("nlist", 100))
            quantizer = (
                self._faiss.IndexFlatIP(self._dimension)
                if self._cfg.metric in {"ip", "cos"}
                else self._faiss.IndexFlatL2(self._dimension)
            )
            index = self._faiss.IndexIVFFlat(quantizer, self._dimension, nlist, metric)
            if vectors.shape[0] >= nlist:
                index.train(vectors)
            else:
                # Keep flat fallback when IVF cannot be trained on too few points.
                index = quantizer
        else:
            index = (
                self._faiss.IndexFlatIP(self._dimension)
                if self._cfg.metric in {"ip", "cos"}
                else self._faiss.IndexFlatL2(self._dimension)
            )

        index.add(vectors)
        return index

    def _finalize_index(self, cpu_index: Any) -> Any:
        return cpu_index

    def _apply_search_params(self, index: Any) -> None:
        nprobe = self._search_params.get("nprobe")
        if nprobe is None:
            nprobe = self._search_params.get("ivf_nprobe")
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)
        ef_search = self._search_params.get("hnsw_ef")
        if ef_search is None:
            ef_search = self._search_params.get("hnsw_ef_search")
        if ef_search is not None and hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(ef_search)

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and int(arr.shape[0]) != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms
