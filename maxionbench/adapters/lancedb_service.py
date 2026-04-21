"""LanceDB service adapter.

Supports either:
1) HTTP mode against a LanceDB service implementing the MaxionBench adapter HTTP contract.
2) Explicit local fallback via `inproc_uri`, which delegates to LanceDbInprocAdapter.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import requests

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from .base import BaseAdapter
from .lancedb_inproc import LanceDbInprocAdapter


class LanceDbServiceAdapter(BaseAdapter):
    """LanceDB service adapter (primary comparable mode)."""

    def __init__(
        self,
        base_url: str | None = "http://127.0.0.1:18080",
        timeout_s: float = 30.0,
        api_key: str | None = None,
        inproc_uri: str | None = None,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._collection = ""
        self._base_url = base_url.rstrip("/") if base_url else None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._delegate = LanceDbInprocAdapter(uri=inproc_uri) if inproc_uri else None

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        if self._delegate is not None:
            self._delegate.create(collection=collection, dimension=dimension, metric=metric)
            return
        self._request(
            "POST",
            "/v1/collections/create",
            json={"collection": collection, "dimension": int(dimension), "metric": metric},
        )

    def drop(self, collection: str) -> None:
        if self._delegate is not None:
            self._delegate.drop(collection=collection)
            if collection == self._collection:
                self._collection = ""
            return
        self._request("DELETE", f"/v1/collections/{collection}", allow_404=True)
        if collection == self._collection:
            self._collection = ""

    def reset(self, collection: str) -> None:
        if self._delegate is not None:
            self._delegate.reset(collection=collection)
            return
        self._request("POST", f"/v1/collections/{collection}/reset")

    def healthcheck(self) -> bool:
        if self._delegate is not None:
            return self._delegate.healthcheck()
        try:
            self._request("GET", "/v1/healthz")
            return True
        except Exception:
            return False

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        if self._delegate is not None:
            return self._delegate.bulk_upsert(records)
        payload = {
            "records": [{"id": r.id, "vector": list(r.vector), "payload": dict(r.payload)} for r in records],
        }
        self._request("POST", f"/v1/collections/{self._collection}/points/upsert", json=payload)
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        if self._delegate is not None:
            return self._delegate.query(request)
        payload = {
            "vector": list(request.vector),
            "top_k": int(request.top_k),
            "filters": dict(request.filters) if request.filters else None,
        }
        body = self._request("POST", f"/v1/collections/{self._collection}/query", json=payload)
        rows = body.get("results") or []
        return [
            QueryResult(
                id=str(item.get("id")),
                score=float(item.get("score", 0.0)),
                payload=dict(item.get("payload") or {}),
            )
            for item in rows
        ]

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        if self._delegate is not None:
            return self._delegate.batch_query(requests)
        payload = {
            "requests": [
                {"vector": list(r.vector), "top_k": int(r.top_k), "filters": dict(r.filters) if r.filters else None}
                for r in requests
            ]
        }
        body = self._request("POST", f"/v1/collections/{self._collection}/batch_query", json=payload)
        batches = body.get("results") or []
        out: list[list[QueryResult]] = []
        for batch in batches:
            out.append(
                [
                    QueryResult(
                        id=str(item.get("id")),
                        score=float(item.get("score", 0.0)),
                        payload=dict(item.get("payload") or {}),
                    )
                    for item in batch
                ]
            )
        return out

    def insert(self, record: UpsertRecord) -> None:
        if self._delegate is not None:
            self._delegate.insert(record)
            return
        self._request(
            "POST",
            f"/v1/collections/{self._collection}/points/insert",
            json={"record": {"id": record.id, "vector": list(record.vector), "payload": dict(record.payload)}},
        )

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        if self._delegate is not None:
            return self._delegate.update_vectors(ids, vectors)
        payload = {"ids": list(ids), "vectors": [list(v) for v in vectors]}
        self._request("POST", f"/v1/collections/{self._collection}/points/update_vectors", json=payload)
        return len(ids)

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        if self._delegate is not None:
            return self._delegate.update_payload(ids, payload)
        self._request(
            "POST",
            f"/v1/collections/{self._collection}/points/update_payload",
            json={"ids": list(ids), "payload": dict(payload)},
        )
        return len(ids)

    def delete(self, ids: Sequence[str]) -> int:
        if self._delegate is not None:
            return self._delegate.delete(ids)
        self._request("POST", f"/v1/collections/{self._collection}/points/delete", json={"ids": list(ids)})
        return len(ids)

    def flush_or_commit(self) -> None:
        if self._delegate is not None:
            self._delegate.flush_or_commit()
            return
        self._request("POST", f"/v1/collections/{self._collection}/flush")

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        if self._delegate is not None:
            self._delegate.set_index_params(params)
            return
        self._request("POST", f"/v1/collections/{self._collection}/set_index_params", json={"params": dict(params)})

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        if self._delegate is not None:
            self._delegate.set_search_params(params)
            return
        self._request("POST", f"/v1/collections/{self._collection}/set_search_params", json={"params": dict(params)})

    def optimize_or_compact(self) -> None:
        if self._delegate is not None:
            self._delegate.optimize_or_compact()
            return
        self._request("POST", f"/v1/collections/{self._collection}/optimize")

    def stats(self) -> AdapterStats:
        if self._delegate is not None:
            return self._delegate.stats()
        body = self._request("GET", f"/v1/collections/{self._collection}/stats")
        return AdapterStats(
            vector_count=int(body.get("vector_count", 0)),
            deleted_count=int(body.get("deleted_count", 0)),
            index_size_bytes=int(body.get("index_size_bytes", 0)),
            ram_usage_bytes=int(body.get("ram_usage_bytes", 0)),
            disk_usage_bytes=int(body.get("disk_usage_bytes", 0)),
            engine_uptime_s=float(body.get("engine_uptime_s", 0.0)),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        if not self._base_url:
            raise RuntimeError("LanceDbServiceAdapter requires base_url when inproc_uri is not set.")
        response = requests.request(
            method=method,
            url=f"{self._base_url}{path}",
            headers=self._headers,
            json=json,
            timeout=self._timeout_s,
        )
        if allow_404 and response.status_code == 404:
            return {}
        response.raise_for_status()
        if not response.content:
            return {}
        body = response.json()
        if isinstance(body, dict) and body.get("status") not in (None, "ok"):
            raise RuntimeError(f"LanceDB service returned non-ok status: {body.get('status')}")
        if not isinstance(body, dict):
            raise RuntimeError("LanceDB service returned non-object JSON response")
        return body
