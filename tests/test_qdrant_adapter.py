from __future__ import annotations

import importlib.util

import pytest
import requests

from maxionbench.adapters.qdrant import QdrantAdapter
from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord


def test_qdrant_metric_mapping() -> None:
    assert QdrantAdapter._distance_name("ip") == "Dot"
    assert QdrantAdapter._distance_name("l2") == "Euclid"
    assert QdrantAdapter._distance_name("cosine") == "Cosine"


def test_qdrant_filter_translation() -> None:
    filt = QdrantAdapter._to_filter({"tenant_id": "a", "acl": 3})
    assert "must" in filt
    assert len(filt["must"]) == 2


def test_qdrant_local_id_roundtrip_helpers() -> None:
    adapter = QdrantAdapter()
    encoded = adapter._encode_local_id("doc-1")
    decoded = adapter._decode_local_id(encoded, {"__maxionbench_id": "doc-1"})
    assert decoded == "doc-1"


def test_qdrant_local_mode_requires_qdrant_client_if_unavailable() -> None:
    if importlib.util.find_spec("qdrant_client") is not None:
        pytest.skip("qdrant-client installed in environment")
    with pytest.raises(ImportError):
        QdrantAdapter(location=":memory:")


def test_qdrant_request_surfaces_http_response_body(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    class _ErrorResponse:
        status_code = 400
        text = '{"status":{"error":"bad payload"}}'
        content = text.encode("utf-8")

        def raise_for_status(self) -> None:
            raise requests.HTTPError("400 Client Error", response=self)

    def _fake_request(**kwargs):  # type: ignore[no-untyped-def]
        return _ErrorResponse()

    monkeypatch.setattr("maxionbench.adapters.qdrant.requests.request", _fake_request)
    adapter = QdrantAdapter()

    with pytest.raises(RuntimeError, match="bad payload"):
        adapter._request("PUT", "/collections/maxionbench/points", json={"points": []})


def test_qdrant_http_bulk_upsert_encodes_string_ids_and_preserves_original_id(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    captured: dict[str, object] = {}

    def _fake_request(method, path, *, json=None, allow_404=False):  # type: ignore[no-untyped-def]
        captured["method"] = method
        captured["path"] = path
        captured["json"] = json
        captured["allow_404"] = allow_404
        return {}

    adapter = QdrantAdapter()
    monkeypatch.setattr(adapter, "_request", _fake_request)

    written = adapter.bulk_upsert([UpsertRecord(id="doc-1", vector=[1.0, 2.0], payload={"tenant": "a"})])

    assert written == 1
    assert captured["method"] == "PUT"
    body = captured["json"]
    assert isinstance(body, dict)
    point = body["points"][0]
    assert point["id"] == adapter._encode_local_id("doc-1")
    assert point["payload"] == {"tenant": "a", "__maxionbench_id": "doc-1"}


def test_qdrant_http_query_decodes_original_id_and_hides_internal_payload(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    adapter = QdrantAdapter()
    encoded_id = adapter._encode_local_id("doc-1")

    def _fake_request(method, path, *, json=None, allow_404=False):  # type: ignore[no-untyped-def]
        return {
            "result": [
                {
                    "id": encoded_id,
                    "score": 0.5,
                    "payload": {"tenant": "a", "__maxionbench_id": "doc-1"},
                }
            ]
        }

    monkeypatch.setattr(adapter, "_request", _fake_request)

    results = adapter.query(QueryRequest(vector=[1.0, 2.0], top_k=1))

    assert len(results) == 1
    assert results[0].id == "doc-1"
    assert results[0].payload == {"tenant": "a"}


def test_qdrant_http_update_and_delete_use_encoded_ids(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def _fake_request(method, path, *, json=None, allow_404=False):  # type: ignore[no-untyped-def]
        calls.append((method, path, json))
        return {}

    adapter = QdrantAdapter()
    monkeypatch.setattr(adapter, "_request", _fake_request)

    adapter.update_vectors(["doc-1"], [[1.0, 2.0]])
    adapter.update_payload(["doc-1"], {"tenant": "a"})
    adapter.delete(["doc-1"])

    encoded_id = adapter._encode_local_id("doc-1")
    assert calls[0][2] == {"points": [{"id": encoded_id, "vector": [1.0, 2.0]}]}
    assert calls[1][2] == {"points": [encoded_id], "payload": {"tenant": "a"}}
    assert calls[2][2] == {"points": [encoded_id]}
