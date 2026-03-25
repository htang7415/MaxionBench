"""Milvus adapter."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from ._exact import StoredPoint, normalize_metric
from .base import BaseAdapter

_FILTERABLE_FIELDS = {"tenant_id", "acl_bucket", "time_bucket"}


class MilvusAdapter(BaseAdapter):
    """Milvus adapter using remote vector search with pinned equality filters."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19530,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        db_name: str | None = None,
        token: str | None = None,
        remote_insert_batch_size: int = 1000,
    ) -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
        except ImportError as exc:
            raise ImportError("pymilvus is required for MilvusAdapter. Install with `pip install pymilvus`.") from exc
        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        self._connections = connections
        self._utility = utility

        self._alias = f"maxionbench-{uuid.uuid4().hex[:8]}"
        if uri:
            self._connections.connect(alias=self._alias, uri=uri, user=user, password=password, db_name=db_name, token=token)
        else:
            self._connections.connect(
                alias=self._alias,
                host=host,
                port=str(port),
                user=user,
                password=password,
                db_name=db_name,
                token=token,
            )

        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}
        self._pending_upserts: dict[str, StoredPoint] = {}
        self._pending_deletes: set[str] = set()
        self._records: dict[str, StoredPoint] = {}
        self._deleted_total = 0
        self._obj: Any | None = None
        self._remote_insert_batch_size = max(1, int(remote_insert_batch_size))

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._dimension = int(dimension)
        self._metric = normalize_metric(metric)
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._records.clear()
        self._deleted_total = 0
        self._create_remote_collection(drop_existing=True)
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        if self._utility.has_collection(collection, using=self._alias):
            self._utility.drop_collection(collection, using=self._alias)
        if collection == self._collection:
            self._collection = ""
            self._dimension = 0
            self._obj = None
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
            return self._connections.has_connection(self._alias)
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
        if self._obj is None:
            return []
        params = self._milvus_search_params()
        search_kwargs: dict[str, Any] = {
            "data": [query_vec.tolist()],
            "anns_field": "vector",
            "param": params,
            "limit": int(request.top_k),
            "output_fields": ["payload_json"],
        }
        expr = self._milvus_expr(request.filters)
        if expr:
            search_kwargs["expr"] = expr
        res = self._obj.search(**search_kwargs)
        hits = res[0] if res else []
        out: list[QueryResult] = []
        for hit in hits:
            payload_raw = hit.entity.get("payload_json") if hasattr(hit, "entity") else None
            payload = self._decode_payload(payload_raw)
            out.append(QueryResult(id=str(hit.id), score=float(hit.score), payload=payload))
        return out

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
        self._sync_remote()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)
        if self._obj is not None:
            self._create_index()

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        if self._obj is not None:
            try:
                self._obj.flush()
            except Exception:
                pass

    def stats(self) -> AdapterStats:
        vector_count = len(self._records)
        index_size = vector_count * self._dimension * 4
        if self._obj is not None:
            try:
                vector_count = int(self._obj.num_entities)
            except Exception:
                pass
        return AdapterStats(
            vector_count=vector_count,
            deleted_count=self._deleted_total,
            index_size_bytes=index_size,
            ram_usage_bytes=0,
            disk_usage_bytes=index_size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _create_index(self) -> None:
        if self._obj is None:
            return
        metric_type = {"ip": "IP", "l2": "L2", "cos": "COSINE"}[self._metric]
        index_type = str(self._index_params.get("index_type", "HNSW")).upper()
        if index_type not in {"HNSW", "IVF_FLAT"}:
            index_type = "HNSW"
        params = {}
        if index_type == "HNSW":
            params = {
                "M": int(self._index_params.get("M", self._index_params.get("hnsw_m", 16))),
                "efConstruction": int(self._index_params.get("efConstruction", self._index_params.get("hnsw_ef_construction", 200))),
            }
        elif index_type == "IVF_FLAT":
            params = {"nlist": int(self._index_params.get("nlist", 1024))}
        try:
            self._obj.create_index(
                field_name="vector",
                index_params={"index_type": index_type, "metric_type": metric_type, "params": params},
            )
        except Exception:
            pass

    def _milvus_search_params(self) -> dict[str, Any]:
        metric_type = {"ip": "IP", "l2": "L2", "cos": "COSINE"}[self._metric]
        params: dict[str, Any] = {}
        if "hnsw_ef" in self._search_params:
            params["ef"] = int(self._search_params["hnsw_ef"])
        if "hnsw_ef_search" in self._search_params:
            params["ef"] = int(self._search_params["hnsw_ef_search"])
        if "nprobe" in self._search_params:
            params["nprobe"] = int(self._search_params["nprobe"])
        return {"metric_type": metric_type, "params": params}

    def _sync_remote(self) -> None:
        snapshot = {
            doc_id: StoredPoint(vector=point.vector.copy(), payload=dict(point.payload))
            for doc_id, point in self._records.items()
        }
        self._create_remote_collection(drop_existing=True)
        if not snapshot:
            return
        if self._obj is None:
            return
        rows = [self._build_remote_row(doc_id=doc_id, point=snapshot[doc_id]) for doc_id in sorted(snapshot.keys())]
        for start in range(0, len(rows), self._remote_insert_batch_size):
            stop = min(len(rows), start + self._remote_insert_batch_size)
            self._obj.insert(rows[start:stop])
        self._obj.flush()
        self._obj.load()

    def _build_remote_row(self, *, doc_id: str, point: StoredPoint) -> dict[str, Any]:
        vector = np.asarray(point.vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Milvus remote row {doc_id!r} has non-1D vector")
        if self._dimension and int(vector.shape[0]) != int(self._dimension):
            raise ValueError(
                f"Milvus remote row {doc_id!r} has vector dimension {int(vector.shape[0])}; expected {int(self._dimension)}"
            )
        payload = dict(point.payload)
        payload_json = json.dumps(payload, sort_keys=True)
        if len(doc_id) > 256:
            raise ValueError(f"Milvus remote row {doc_id!r} exceeds id max_length=256")
        tenant_id = str(payload.get("tenant_id", ""))
        if len(tenant_id) > 256:
            raise ValueError(f"Milvus remote row {doc_id!r} exceeds tenant_id max_length=256")
        if len(payload_json) > 65535:
            raise ValueError(f"Milvus remote row {doc_id!r} exceeds payload_json max_length=65535")
        return {
            "id": doc_id,
            "vector": vector.tolist(),
            "payload_json": payload_json,
            "tenant_id": tenant_id,
            "acl_bucket": int(payload.get("acl_bucket", 0)),
            "time_bucket": int(payload.get("time_bucket", 0)),
        }

    def _create_remote_collection(self, *, drop_existing: bool) -> None:
        if drop_existing and self._utility.has_collection(self._collection, using=self._alias):
            self._utility.drop_collection(self._collection, using=self._alias)
        fields = [
            self._FieldSchema(name="id", dtype=self._DataType.VARCHAR, is_primary=True, max_length=256),
            self._FieldSchema(name="vector", dtype=self._DataType.FLOAT_VECTOR, dim=self._dimension),
            self._FieldSchema(name="payload_json", dtype=self._DataType.VARCHAR, max_length=65535),
            self._FieldSchema(name="tenant_id", dtype=self._DataType.VARCHAR, max_length=256),
            self._FieldSchema(name="acl_bucket", dtype=self._DataType.INT64),
            self._FieldSchema(name="time_bucket", dtype=self._DataType.INT64),
        ]
        schema = self._CollectionSchema(fields=fields, description="maxionbench milvus collection")
        self._obj = self._Collection(name=self._collection, schema=schema, using=self._alias)
        self._create_index()
        self._obj.load()

    def _decode_payload(self, payload_raw: Any) -> dict[str, Any]:
        if payload_raw is None:
            return {}
        if isinstance(payload_raw, dict):
            return dict(payload_raw)
        try:
            parsed = json.loads(str(payload_raw))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    def __del__(self) -> None:
        try:
            self._connections.disconnect(alias=self._alias)
        except Exception:
            pass

    def _milvus_expr(self, filters: Mapping[str, Any] | None) -> str:
        if not filters:
            return ""
        clauses: list[str] = []
        for key, value in sorted(filters.items()):
            key_text = str(key)
            if key_text not in _FILTERABLE_FIELDS:
                raise ValueError(
                    f"MilvusAdapter only supports equality filters on {sorted(_FILTERABLE_FIELDS)}; got {key_text!r}"
                )
            clauses.append(f"{key_text} == {json.dumps(value)}")
        return " and ".join(clauses)
