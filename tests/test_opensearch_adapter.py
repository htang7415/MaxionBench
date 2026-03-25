from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import requests

from maxionbench.adapters._exact import StoredPoint
from maxionbench.adapters.opensearch import OpenSearchAdapter


class _OkBulkResponse:
    status_code = 200
    content = b"{}"
    text = "{}"

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return {"errors": False}


class _HttpErrorResponse:
    def __init__(self, *, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self) -> dict[str, Any]:
        return {"error": self.text}


def test_opensearch_rewrite_remote_index_chunks_bulk_payloads(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    adapter = OpenSearchAdapter(bulk_max_records=2, bulk_max_bytes=4096)
    adapter._index = "collection"
    adapter._dimension = 2
    adapter._metric = "ip"
    adapter._records = {
        "doc-0": StoredPoint(
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            payload={"text": "seed", "rank": 0},
        )
    }
    adapter._pending_deletes = {"doc-0"}
    adapter._pending_upserts = {
        f"doc-{idx}": StoredPoint(
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            payload={"text": "x" * 220, "rank": idx},
        )
        for idx in range(1, 4)
    }

    request_calls: list[tuple[str, str, str]] = []
    bulk_payloads: list[str] = []

    def _fake_session_request(method: str, url: str, **kwargs: Any) -> _OkBulkResponse:
        request_calls.append((method, url, str(kwargs.get("headers", {}).get("Content-Type", ""))))
        bulk_payloads.append(str(kwargs["data"]))
        return _OkBulkResponse()

    monkeypatch.setattr(adapter._session, "request", _fake_session_request)

    adapter.flush_or_commit()

    assert len(bulk_payloads) == 2
    assert all(payload.endswith("\n") for payload in bulk_payloads)
    assert all(method == "POST" for method, _, _ in request_calls)
    assert all(url == "http://127.0.0.1:9200/_bulk?refresh=wait_for" for _, url, _ in request_calls)
    assert all(content_type == "application/x-ndjson" for _, _, content_type in request_calls)
    assert any('"delete"' in payload for payload in bulk_payloads)
    assert not adapter._pending_upserts
    assert not adapter._pending_deletes
    assert set(adapter._records.keys()) == {"doc-1", "doc-2", "doc-3"}


def test_opensearch_request_surfaces_http_response_body() -> None:
    response = _HttpErrorResponse(status_code=413, text='{"error":"payload too large"}')

    with pytest.raises(RuntimeError, match="payload too large"):
        OpenSearchAdapter._raise_for_status_with_body(response, context="OpenSearch bulk request failed")


def test_opensearch_request_uses_pooled_session(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    adapter = OpenSearchAdapter()
    captured: dict[str, object] = {}

    class _OkResponse:
        status_code = 200
        content = b"{}"
        text = "{}"

        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {"ok": True}

    def _fake_request(method: str, url: str, **kwargs: Any) -> _OkResponse:
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = kwargs.get("timeout")
        return _OkResponse()

    monkeypatch.setattr(adapter._session, "request", _fake_request)

    payload = adapter._request("GET", "/")

    assert payload == {"ok": True}
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:9200/"
