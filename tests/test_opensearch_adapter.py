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
    adapter = OpenSearchAdapter(bulk_max_records=2, bulk_max_bytes=512)
    adapter._index = "collection"
    adapter._dimension = 2
    adapter._metric = "ip"
    adapter._records = {
        f"doc-{idx}": StoredPoint(
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            payload={"text": "x" * 220, "rank": idx},
        )
        for idx in range(3)
    }

    request_calls: list[tuple[str, str]] = []
    bulk_payloads: list[str] = []

    def _fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        request_calls.append((method, path))
        return {}

    def _fake_post(url: str, **kwargs: Any) -> _OkBulkResponse:
        bulk_payloads.append(str(kwargs["data"]))
        return _OkBulkResponse()

    monkeypatch.setattr(adapter, "_request", _fake_request)
    monkeypatch.setattr("maxionbench.adapters.opensearch.requests.post", _fake_post)

    adapter._rewrite_remote_index()

    assert len(bulk_payloads) == 2
    assert all(payload.endswith("\n") for payload in bulk_payloads)
    assert request_calls[0] == ("DELETE", "/collection")
    assert request_calls[1] == ("PUT", "/collection")
    assert request_calls[-1] == ("POST", "/collection/_refresh")


def test_opensearch_request_surfaces_http_response_body() -> None:
    response = _HttpErrorResponse(status_code=413, text='{"error":"payload too large"}')

    with pytest.raises(RuntimeError, match="payload too large"):
        OpenSearchAdapter._raise_for_status_with_body(response, context="OpenSearch bulk request failed")
