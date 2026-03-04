"""Qdrant adapter implemented via HTTP API.

This adapter works against an already running Qdrant service.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence
import uuid

import requests

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from .base import BaseAdapter


@dataclass(frozen=True)
class _QdrantConfig:
    host: str = "127.0.0.1"
    port: int = 6333
    api_key: str | None = None
    timeout_s: float = 30.0
    location: str | None = None


class QdrantAdapter(BaseAdapter):
    """Qdrant adapter with required MaxionBench contract methods."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        location: str | None = None,
    ) -> None:
        self._cfg = _QdrantConfig(
            host=host,
            port=port,
            api_key=api_key,
            timeout_s=timeout_s,
            location=location,
        )
        self._base_url = f"http://{self._cfg.host}:{self._cfg.port}"
        self._headers = {"api-key": api_key} if api_key else {}
        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._search_params: dict[str, Any] = {}
        self._index_params: dict[str, Any] = {}
        self._local_client: Any | None = None
        self._qm: Any | None = None
        if self._cfg.location is not None:
            self._init_local_client()

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._dimension = dimension
        self._metric = metric
        distance = self._distance_name(metric)
        if self._local_client is not None:
            self._local_client.create_collection(
                collection_name=collection,
                vectors_config=self._qm.VectorParams(
                    size=dimension,
                    distance=getattr(self._qm.Distance, distance.upper()),
                ),
            )
            self._created_at = time.monotonic()
            return
        body = {
            "vectors": {
                "size": dimension,
                "distance": distance,
            }
        }
        if self._index_params:
            body.update(self._index_params)
        self._request("PUT", f"/collections/{collection}", json=body)
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        if self._local_client is not None:
            if self._local_client.collection_exists(collection):
                self._local_client.delete_collection(collection_name=collection)
            if collection == self._collection:
                self._collection = ""
            return
        self._request("DELETE", f"/collections/{collection}", allow_404=True)
        if collection == self._collection:
            self._collection = ""

    def reset(self, collection: str) -> None:
        dimension = self._dimension or 1
        metric = self._metric
        self.drop(collection)
        self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        if self._local_client is not None:
            return True
        try:
            self._request("GET", "/collections")
            return True
        except Exception:
            return False

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        if self._local_client is not None:
            points = [
                self._qm.PointStruct(
                    id=self._encode_local_id(record.id),
                    vector=list(record.vector),
                    payload=self._with_local_id_payload(record.id, record.payload),
                )
                for record in records
            ]
            self._local_client.upsert(collection_name=self._collection, points=points, wait=True)
            return len(records)
        points = [
            {
                "id": record.id,
                "vector": list(record.vector),
                "payload": dict(record.payload),
            }
            for record in records
        ]
        body = {"points": points}
        self._request("PUT", f"/collections/{self._collection}/points?wait=true", json=body)
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        if self._local_client is not None:
            query_filter = self._to_local_filter(request.filters) if request.filters else None
            search_params = self._qm.SearchParams(**self._search_params) if self._search_params else None
            response = self._local_client.query_points(
                collection_name=self._collection,
                query=list(request.vector),
                query_filter=query_filter,
                search_params=search_params,
                limit=request.top_k,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points
            return [
                QueryResult(
                    id=self._decode_local_id(point.id, point.payload),
                    score=float(point.score),
                    payload=self._without_local_id_payload(point.payload),
                )
                for point in points
            ]
        body: dict[str, Any] = {
            "vector": list(request.vector),
            "limit": request.top_k,
            "with_payload": True,
        }
        if self._search_params:
            body["params"] = dict(self._search_params)
        if request.filters:
            body["filter"] = self._to_filter(request.filters)
        payload = self._request("POST", f"/collections/{self._collection}/points/search", json=body)
        results = payload.get("result", [])
        return [
            QueryResult(
                id=str(item.get("id")),
                score=float(item.get("score", 0.0)),
                payload=dict(item.get("payload") or {}),
            )
            for item in results
        ]

    def batch_query(self, requests_batch: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests_batch]

    def insert(self, record: UpsertRecord) -> None:
        self.bulk_upsert([record])

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if self._local_client is not None:
            points = [
                self._qm.PointVectors(id=self._encode_local_id(doc_id), vector=list(vector))
                for doc_id, vector in zip(ids, vectors)
            ]
            self._local_client.update_vectors(collection_name=self._collection, points=points, wait=True)
            return len(points)
        points = [{"id": doc_id, "vector": list(vector)} for doc_id, vector in zip(ids, vectors)]
        self._request(
            "PUT",
            f"/collections/{self._collection}/points/vectors?wait=true",
            json={"points": points},
        )
        return len(points)

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        if self._local_client is not None:
            for doc_id in ids:
                self._local_client.set_payload(
                    collection_name=self._collection,
                    points=[self._encode_local_id(doc_id)],
                    payload=dict(payload),
                    wait=True,
                )
            return len(ids)
        self._request(
            "POST",
            f"/collections/{self._collection}/points/payload?wait=true",
            json={"points": list(ids), "payload": dict(payload)},
        )
        return len(ids)

    def delete(self, ids: Sequence[str]) -> int:
        if self._local_client is not None:
            encoded_ids = [self._encode_local_id(doc_id) for doc_id in ids]
            self._local_client.delete(collection_name=self._collection, points_selector=encoded_ids, wait=True)
            return len(ids)
        self._request(
            "POST",
            f"/collections/{self._collection}/points/delete?wait=true",
            json={"points": list(ids)},
        )
        return len(ids)

    def flush_or_commit(self) -> None:
        # Qdrant writes are acknowledged with wait=true in this adapter path.
        return

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        # Qdrant compaction is managed internally; exposing as explicit no-op.
        return

    def stats(self) -> AdapterStats:
        if self._local_client is not None:
            info = self._local_client.get_collection(collection_name=self._collection)
            points_count = int(info.points_count or info.vectors_count or 0)
            return AdapterStats(
                vector_count=points_count,
                deleted_count=0,
                index_size_bytes=0,
                ram_usage_bytes=0,
                disk_usage_bytes=0,
                engine_uptime_s=time.monotonic() - self._created_at,
            )
        payload = self._request("GET", f"/collections/{self._collection}")
        result = payload.get("result", {})
        points_count = int(result.get("points_count") or result.get("vectors_count") or 0)
        segments = result.get("segments_count") or 0
        # Qdrant does not always expose all sizes in the same fields; keep robust defaults.
        disk_usage_bytes = int(result.get("disk_data_size") or 0)
        ram_usage_bytes = int(result.get("ram_data_size") or 0)
        return AdapterStats(
            vector_count=points_count,
            deleted_count=0,
            index_size_bytes=disk_usage_bytes if disk_usage_bytes > 0 else segments * self._dimension * 4,
            ram_usage_bytes=ram_usage_bytes,
            disk_usage_bytes=disk_usage_bytes,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        response = requests.request(
            method=method,
            url=url,
            headers=self._headers,
            json=json,
            timeout=self._cfg.timeout_s,
        )
        if allow_404 and response.status_code == 404:
            return {}
        response.raise_for_status()
        if not response.content:
            return {}
        body = response.json()
        status = body.get("status")
        if status not in (None, "ok"):
            raise RuntimeError(f"Qdrant returned non-ok status: {status}")
        return body

    @staticmethod
    def _distance_name(metric: str) -> str:
        normalized = metric.strip().lower()
        if normalized in ("ip", "dot", "inner_product"):
            return "Dot"
        if normalized in ("l2", "euclid", "euclidean"):
            return "Euclid"
        if normalized in ("cos", "cosine"):
            return "Cosine"
        raise ValueError(f"Unsupported metric for Qdrant: {metric}")

    @staticmethod
    def _to_filter(filters: Mapping[str, Any]) -> dict[str, Any]:
        must = [{"key": key, "match": {"value": value}} for key, value in filters.items()]
        return {"must": must}

    def _init_local_client(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for local-mode QdrantAdapter. "
                "Install with `pip install qdrant-client`."
            ) from exc
        self._local_client = QdrantClient(location=self._cfg.location)
        self._qm = qm

    def _to_local_filter(self, filters: Mapping[str, Any]) -> Any:
        must = [
            self._qm.FieldCondition(
                key=key,
                match=self._qm.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return self._qm.Filter(must=must)

    @staticmethod
    def _with_local_id_payload(doc_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(payload)
        merged["__maxionbench_id"] = doc_id
        return merged

    @staticmethod
    def _without_local_id_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        cleaned = dict(payload or {})
        cleaned.pop("__maxionbench_id", None)
        return cleaned

    def _encode_local_id(self, doc_id: str) -> str:
        try:
            parsed = uuid.UUID(str(doc_id))
            return str(parsed)
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

    def _decode_local_id(self, point_id: Any, payload: Mapping[str, Any] | None) -> str:
        payload_map = dict(payload or {})
        if "__maxionbench_id" in payload_map:
            return str(payload_map["__maxionbench_id"])
        return str(point_id)
