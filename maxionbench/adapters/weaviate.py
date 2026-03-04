"""Weaviate adapter."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import requests

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from ._exact import StoredPoint, normalize_metric, topk_exact
from .base import BaseAdapter


class WeaviateAdapter(BaseAdapter):
    """Weaviate adapter with deterministic exact local query path."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, scheme: str = "http", timeout_s: float = 30.0) -> None:
        self._base_url = f"{scheme}://{host}:{port}"
        self._timeout_s = float(timeout_s)
        self._collection = ""
        self._class_name = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}
        self._records: dict[str, StoredPoint] = {}
        self._pending_upserts: dict[str, StoredPoint] = {}
        self._pending_deletes: set[str] = set()
        self._deleted_total = 0

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._class_name = self._to_class_name(collection)
        self._dimension = int(dimension)
        self._metric = normalize_metric(metric)
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._create_remote_class()
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        class_name = self._to_class_name(collection)
        self._request("DELETE", f"/v1/schema/{class_name}", allow_404=True)
        if collection == self._collection:
            self._collection = ""
            self._class_name = ""
            self._records.clear()
            self._pending_upserts.clear()
            self._pending_deletes.clear()
            self._deleted_total = 0

    def reset(self, collection: str) -> None:
        dimension = self._dimension or 1
        metric = self._metric
        self.drop(collection)
        self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        try:
            self._request("GET", "/v1/.well-known/ready")
            return True
        except Exception:
            return False

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        for record in records:
            key = str(record.id)
            self._pending_upserts[key] = StoredPoint(vector=self._to_vector(record.vector), payload=dict(record.payload))
            self._pending_deletes.discard(key)
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        return topk_exact(
            records=self._records,
            query=self._to_vector(request.vector),
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
        self._rewrite_remote_objects()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        # Weaviate compaction is managed internally.
        return

    def stats(self) -> AdapterStats:
        vector_count = len(self._records)
        size = vector_count * self._dimension * 4 + (vector_count * 80)
        return AdapterStats(
            vector_count=vector_count,
            deleted_count=self._deleted_total,
            index_size_bytes=size,
            ram_usage_bytes=0,
            disk_usage_bytes=size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _create_remote_class(self) -> None:
        self._request("DELETE", f"/v1/schema/{self._class_name}", allow_404=True)
        vector_distance = {"ip": "dot", "l2": "l2-squared", "cos": "cosine"}[self._metric]
        body = {
            "class": self._class_name,
            "vectorizer": "none",
            "vectorIndexConfig": {"distance": vector_distance},
            "properties": [
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "payload_json", "dataType": ["text"]},
            ],
        }
        self._request("POST", "/v1/schema", json=body)

    def _rewrite_remote_objects(self) -> None:
        self._create_remote_class()
        if not self._records:
            return
        objects = []
        for doc_id, point in sorted(self._records.items(), key=lambda item: item[0]):
            objects.append(
                {
                    "class": self._class_name,
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id)),
                    "vector": point.vector.tolist(),
                    "properties": {
                        "doc_id": doc_id,
                        "payload_json": json.dumps(point.payload, sort_keys=True),
                    },
                }
            )
        self._request("POST", "/v1/batch/objects", json={"objects": objects})

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        response = requests.request(
            method=method,
            url=f"{self._base_url}{path}",
            json=json,
            timeout=self._timeout_s,
        )
        if allow_404 and response.status_code == 404:
            return {}
        response.raise_for_status()
        if not response.content:
            return {}
        body = response.json()
        if not isinstance(body, dict):
            return {}
        return body

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _to_class_name(collection: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in collection)
        if not cleaned:
            return "MaxionbenchCollection"
        return cleaned[0].upper() + cleaned[1:]
