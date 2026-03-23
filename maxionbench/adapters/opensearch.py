"""OpenSearch k-NN adapter."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

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


class OpenSearchAdapter(BaseAdapter):
    """OpenSearch k-NN adapter with exact local fallback for filtered queries."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9200,
        scheme: str = "http",
        username: str | None = None,
        password: str | None = None,
        timeout_s: float = 30.0,
        healthcheck_timeout_s: float | None = None,
        verify_ssl: bool = False,
        bulk_max_records: int = 1000,
        bulk_max_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self._base_url = f"{scheme}://{host}:{port}"
        self._timeout_s = float(timeout_s)
        self._healthcheck_timeout_s = float(healthcheck_timeout_s) if healthcheck_timeout_s is not None else self._timeout_s
        self._verify_ssl = bool(verify_ssl)
        self._auth = (username, password) if username and password else None
        self._bulk_max_records = max(1, int(bulk_max_records))
        self._bulk_max_bytes = max(1024, int(bulk_max_bytes))

        self._index = ""
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
        self._index = collection
        self._dimension = int(dimension)
        self._metric = normalize_metric(metric)
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._create_remote_index()
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        self._request("DELETE", f"/{collection}", allow_404=True)
        if collection == self._index:
            self._index = ""
            self._dimension = 0
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
            self._request("GET", "/", timeout_s=self._healthcheck_timeout_s)
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
        query_vec = self._to_vector(request.vector)
        if request.filters:
            return topk_exact(
                records=self._records,
                query=query_vec,
                top_k=request.top_k,
                metric=self._metric,
                filters=request.filters,
            )
        try:
            body = {
                "size": int(request.top_k),
                "_source": ["payload"],
                "query": {
                    "knn": {
                        "vector": {
                            "vector": query_vec.tolist(),
                            "k": int(request.top_k),
                        }
                    }
                },
            }
            response = self._request("POST", f"/{self._index}/_search", json=body)
            hits = (((response.get("hits") or {}).get("hits")) or [])
            out: list[QueryResult] = []
            for hit in hits:
                source = hit.get("_source") or {}
                payload = dict(source.get("payload") or {})
                out.append(
                    QueryResult(
                        id=str(hit.get("_id")),
                        score=float(hit.get("_score", 0.0)),
                        payload=payload,
                    )
                )
            return out
        except Exception:
            return topk_exact(
                records=self._records,
                query=query_vec,
                top_k=request.top_k,
                metric=self._metric,
                filters=None,
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
        self._rewrite_remote_index()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        try:
            self._request("POST", f"/{self._index}/_forcemerge?max_num_segments=1")
        except Exception:
            pass

    def stats(self) -> AdapterStats:
        vector_count = len(self._records)
        index_size = vector_count * self._dimension * 4
        disk_size = index_size
        try:
            body = self._request("GET", f"/{self._index}/_stats/store,docs")
            indices = (((body.get("indices") or {}).get(self._index)) or {})
            primaries = indices.get("primaries") or {}
            store = primaries.get("store") or {}
            docs = primaries.get("docs") or {}
            disk_size = int(store.get("size_in_bytes", disk_size))
            vector_count = int(docs.get("count", vector_count))
        except Exception:
            pass
        return AdapterStats(
            vector_count=vector_count,
            deleted_count=self._deleted_total,
            index_size_bytes=disk_size,
            ram_usage_bytes=0,
            disk_usage_bytes=disk_size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _create_remote_index(self) -> None:
        self._request("DELETE", f"/{self._index}", allow_404=True)
        space_type = {"ip": "innerproduct", "l2": "l2", "cos": "cosinesimil"}[self._metric]
        method = {
            "name": "hnsw",
            "engine": str(self._index_params.get("engine", "nmslib")),
            "space_type": space_type,
            "parameters": {
                "ef_construction": int(self._index_params.get("ef_construction", 128)),
                "m": int(self._index_params.get("m", 16)),
            },
        }
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector": {"type": "knn_vector", "dimension": self._dimension, "method": method},
                    "payload": {"type": "object", "enabled": True},
                }
            },
        }
        self._request("PUT", f"/{self._index}", json=body)

    def _rewrite_remote_index(self) -> None:
        self._create_remote_index()
        if not self._records:
            self._request("POST", f"/{self._index}/_refresh")
            return
        lines: list[str] = []
        payload_bytes = 0
        record_count = 0
        for doc_id, point in sorted(self._records.items(), key=lambda item: item[0]):
            meta_line = json.dumps({"index": {"_index": self._index, "_id": doc_id}}, sort_keys=True)
            doc_line = json.dumps({"vector": point.vector.tolist(), "payload": point.payload}, sort_keys=True)
            entry_bytes = len(meta_line.encode("utf-8")) + len(doc_line.encode("utf-8")) + 2
            if lines and (
                record_count >= self._bulk_max_records or payload_bytes + entry_bytes > self._bulk_max_bytes
            ):
                self._post_bulk_lines(lines)
                lines = []
                payload_bytes = 0
                record_count = 0
            lines.extend([meta_line, doc_line])
            payload_bytes += entry_bytes
            record_count += 1
        if lines:
            self._post_bulk_lines(lines)
        self._request("POST", f"/{self._index}/_refresh")

    def _post_bulk_lines(self, lines: Sequence[str]) -> None:
        payload = "\n".join(lines) + "\n"
        response = requests.post(
            f"{self._base_url}/_bulk",
            data=payload,
            headers={"Content-Type": "application/x-ndjson"},
            auth=self._auth,
            timeout=self._timeout_s,
            verify=self._verify_ssl,
        )
        self._raise_for_status_with_body(response, context="OpenSearch bulk request failed")
        body = response.json()
        if bool(body.get("errors")):
            raise RuntimeError(f"OpenSearch bulk operation reported errors: {body}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        allow_404: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        response = requests.request(
            method=method,
            url=f"{self._base_url}{path}",
            json=json,
            auth=self._auth,
            timeout=self._timeout_s if timeout_s is None else float(timeout_s),
            verify=self._verify_ssl,
        )
        if allow_404 and response.status_code == 404:
            return {}
        self._raise_for_status_with_body(response, context=f"OpenSearch {method} {path} failed")
        if not response.content:
            return {}
        body = response.json()
        if isinstance(body, dict):
            return body
        return {}

    @staticmethod
    def _raise_for_status_with_body(response: requests.Response, *, context: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body_preview = OpenSearchAdapter._response_body_preview(response)
            message = context
            if body_preview:
                message = f"{message}: {body_preview}"
            raise RuntimeError(message) from exc

    @staticmethod
    def _response_body_preview(response: requests.Response) -> str:
        text = getattr(response, "text", "")
        if text:
            return text[:1000]
        content = getattr(response, "content", b"") or b""
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")[:1000]
        return str(content)[:1000]

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr
