from __future__ import annotations

import json

import pytest

from maxionbench.adapters.milvus import MilvusAdapter
from maxionbench.adapters.weaviate import WeaviateAdapter
from maxionbench.schemas.adapter_contract import QueryRequest


def test_weaviate_create_materializes_filter_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = WeaviateAdapter()
    captured: dict[str, object] = {}

    def _fake_request(method: str, path: str, *, json=None, allow_404: bool = False):  # type: ignore[no-untyped-def]
        if method == "POST" and path == "/v1/schema":
            captured["body"] = json
        return {}

    monkeypatch.setattr(adapter, "_request", _fake_request)
    adapter.create(collection="bench", dimension=4, metric="ip")
    body = captured["body"]
    assert isinstance(body, dict)
    props = body["properties"]
    names = {item["name"] for item in props}
    assert {"doc_id", "payload_json", "tenant_id", "acl_bucket", "time_bucket"} <= names


def test_weaviate_query_uses_graphql_remote_search(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = WeaviateAdapter()
    adapter._class_name = "Bench"
    adapter._dimension = 2
    adapter._metric = "ip"
    captured: dict[str, object] = {}

    def _fake_request(method: str, path: str, *, json=None, allow_404: bool = False):  # type: ignore[no-untyped-def]
        captured["method"] = method
        captured["path"] = path
        captured["body"] = json
        return {
            "data": {
                "Get": {
                    "Bench": [
                        {
                            "doc_id": "doc-1",
                            "payload_json": json_module.dumps({"tenant_id": "tenant-001"}, sort_keys=True),
                            "_additional": {"distance": 0.25, "id": "uuid-1"},
                        }
                    ]
                }
            }
        }

    json_module = json
    monkeypatch.setattr(adapter, "_request", _fake_request)
    rows = adapter.query(
        QueryRequest(vector=[1.0, 0.0], top_k=5, filters={"tenant_id": "tenant-001", "acl_bucket": 2})
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/graphql"
    body = captured["body"]
    assert isinstance(body, dict)
    graphql = str(body["query"])
    assert "nearVector" in graphql
    assert "tenant_id" in graphql
    assert "acl_bucket" in graphql
    assert rows[0].id == "doc-1"
    assert rows[0].payload["tenant_id"] == "tenant-001"


def test_weaviate_query_rejects_unsupported_filter_key() -> None:
    adapter = WeaviateAdapter()
    adapter._class_name = "Bench"
    adapter._dimension = 2
    with pytest.raises(ValueError, match="tenant_id"):
        adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=3, filters={"region": "us"}))


def test_weaviate_query_surfaces_remote_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = WeaviateAdapter()
    adapter._class_name = "Bench"
    adapter._dimension = 2

    def _raise_request(method: str, path: str, *, json=None, allow_404: bool = False):  # type: ignore[no-untyped-def]
        raise RuntimeError("remote failure")

    monkeypatch.setattr(adapter, "_request", _raise_request)
    with pytest.raises(RuntimeError, match="remote failure"):
        adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=3))


class _FakeMilvusHit:
    def __init__(self, doc_id: str, score: float, payload: dict[str, object]) -> None:
        self.id = doc_id
        self.score = score
        self.entity = {"payload_json": json.dumps(payload, sort_keys=True)}


class _FakeMilvusCollection:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None
        self.loaded = False
        self.schema = None

    def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.search_kwargs = dict(kwargs)
        return [[_FakeMilvusHit("doc-1", 0.9, {"tenant_id": "tenant-001"})]]

    def load(self) -> None:
        self.loaded = True

    def create_index(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None


def _make_milvus_adapter() -> MilvusAdapter:
    adapter = object.__new__(MilvusAdapter)
    adapter._dimension = 2
    adapter._metric = "ip"
    adapter._search_params = {}
    adapter._index_params = {}
    adapter._collection = "bench"
    adapter._alias = "alias"
    adapter._utility = type("Utility", (), {"has_collection": staticmethod(lambda *args, **kwargs: False), "drop_collection": staticmethod(lambda *args, **kwargs: None)})()
    adapter._DataType = type("DataType", (), {"VARCHAR": "VARCHAR", "FLOAT_VECTOR": "FLOAT_VECTOR", "INT64": "INT64"})
    adapter._FieldSchema = lambda **kwargs: dict(kwargs)  # type: ignore[assignment]
    adapter._CollectionSchema = lambda fields, description: {"fields": fields, "description": description}  # type: ignore[assignment]
    adapter._connections = type("Connections", (), {"disconnect": staticmethod(lambda alias: None)})()
    return adapter


def test_milvus_create_remote_collection_materializes_filter_fields() -> None:
    adapter = _make_milvus_adapter()

    collection = _FakeMilvusCollection()

    def _collection_ctor(*, name, schema, using):  # type: ignore[no-untyped-def]
        collection.schema = schema
        return collection

    adapter._Collection = _collection_ctor  # type: ignore[assignment]
    adapter._obj = None
    adapter._create_remote_collection(drop_existing=True)
    fields = collection.schema["fields"]
    names = {item["name"] for item in fields}
    assert {"id", "vector", "payload_json", "tenant_id", "acl_bucket", "time_bucket"} <= names


def test_milvus_query_uses_remote_expr_search() -> None:
    adapter = _make_milvus_adapter()
    collection = _FakeMilvusCollection()
    adapter._obj = collection
    rows = adapter.query(
        QueryRequest(vector=[1.0, 0.0], top_k=5, filters={"tenant_id": "tenant-001", "acl_bucket": 2})
    )
    assert collection.search_kwargs is not None
    assert collection.search_kwargs["expr"] == 'acl_bucket == 2 and tenant_id == "tenant-001"'
    assert rows[0].id == "doc-1"
    assert rows[0].payload["tenant_id"] == "tenant-001"


def test_milvus_query_rejects_unsupported_filter_key() -> None:
    adapter = _make_milvus_adapter()
    adapter._obj = _FakeMilvusCollection()
    with pytest.raises(ValueError, match="tenant_id"):
        adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=3, filters={"region": "us"}))


def test_milvus_query_surfaces_remote_errors() -> None:
    adapter = _make_milvus_adapter()

    class _RaisingCollection:
        def search(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("remote failure")

    adapter._obj = _RaisingCollection()
    with pytest.raises(RuntimeError, match="remote failure"):
        adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=3))
