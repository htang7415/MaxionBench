from __future__ import annotations

import json

import numpy as np
import pytest

from maxionbench.adapters._exact import StoredPoint
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


def test_weaviate_flush_or_commit_chunks_large_remote_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = WeaviateAdapter(batch_max_objects=2)
    adapter._collection = "bench"
    adapter._class_name = "Bench"
    adapter._dimension = 2
    adapter._metric = "ip"
    adapter._pending_upserts = {
        f"doc-{idx}": StoredPoint(
            vector=np.asarray([1.0, 0.0], dtype=np.float32),
            payload={"tenant_id": f"tenant-{idx:03d}", "acl_bucket": idx, "time_bucket": idx},
        )
        for idx in range(3)
    }

    payloads: list[dict[str, object]] = []

    def _fake_request(method: str, path: str, *, json=None, allow_404: bool = False, timeout_s=None):  # type: ignore[no-untyped-def]
        del allow_404, timeout_s
        if method == "POST" and path == "/v1/batch/objects":
            payloads.append(dict(json or {}))
        return {}

    monkeypatch.setattr(adapter, "_request", _fake_request)

    adapter.flush_or_commit()

    assert [len(payload["objects"]) for payload in payloads] == [2, 1]
    assert not adapter._pending_upserts
    assert set(adapter._records.keys()) == {"doc-0", "doc-1", "doc-2"}


class _FakeMilvusHit:
    def __init__(self, doc_id: str, score: float, payload: dict[str, object]) -> None:
        self.id = doc_id
        self.score = score
        self.entity = {"payload_json": json.dumps(payload, sort_keys=True)}


class _FakeMilvusCollection:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None
        self.insert_rows: list[dict[str, object]] | None = None
        self.insert_batches: list[list[dict[str, object]]] = []
        self.flushed = False
        self.loaded = False
        self.schema = None

    def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.search_kwargs = dict(kwargs)
        return [[_FakeMilvusHit("doc-1", 0.9, {"tenant_id": "tenant-001"})]]

    def insert(self, rows):  # type: ignore[no-untyped-def]
        self.insert_rows = list(rows)
        self.insert_batches.append(list(rows))

    def flush(self) -> None:
        self.flushed = True

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
    adapter._remote_insert_batch_size = 1000
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


def test_milvus_query_clamps_hnsw_ef_to_top_k() -> None:
    adapter = _make_milvus_adapter()
    collection = _FakeMilvusCollection()
    adapter._obj = collection
    adapter._search_params = {"hnsw_ef": 32}

    adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=200))

    assert collection.search_kwargs is not None
    payload = collection.search_kwargs["param"]
    assert payload["params"]["ef"] == 200


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


def test_milvus_sync_remote_inserts_row_dicts_in_sorted_id_order() -> None:
    adapter = _make_milvus_adapter()
    collection = _FakeMilvusCollection()
    adapter._records = {
        "doc-2": StoredPoint(
            vector=np.asarray([0.1, 0.9], dtype=np.float32),
            payload={"tenant_id": "tenant-002", "acl_bucket": 1, "time_bucket": 17},
        ),
        "doc-1": StoredPoint(
            vector=np.asarray([0.9, 0.1], dtype=np.float32),
            payload={"tenant_id": "tenant-001", "acl_bucket": 3, "time_bucket": 11},
        ),
    }

    def _create_remote_collection(*, drop_existing: bool) -> None:
        assert drop_existing is True
        adapter._obj = collection

    adapter._create_remote_collection = _create_remote_collection  # type: ignore[method-assign]
    adapter._sync_remote()

    assert collection.insert_rows == [
        {
            "id": "doc-1",
            "vector": [0.8999999761581421, 0.10000000149011612],
            "payload_json": '{"acl_bucket": 3, "tenant_id": "tenant-001", "time_bucket": 11}',
            "tenant_id": "tenant-001",
            "acl_bucket": 3,
            "time_bucket": 11,
        },
        {
            "id": "doc-2",
            "vector": [0.10000000149011612, 0.8999999761581421],
            "payload_json": '{"acl_bucket": 1, "tenant_id": "tenant-002", "time_bucket": 17}',
            "tenant_id": "tenant-002",
            "acl_bucket": 1,
            "time_bucket": 17,
        },
    ]
    assert collection.flushed is True
    assert collection.loaded is True


def test_milvus_sync_remote_batches_large_snapshot() -> None:
    adapter = _make_milvus_adapter()
    adapter._remote_insert_batch_size = 1
    collection = _FakeMilvusCollection()
    adapter._records = {
        "doc-2": StoredPoint(
            vector=np.asarray([0.1, 0.9], dtype=np.float32),
            payload={"tenant_id": "tenant-002", "acl_bucket": 1, "time_bucket": 17},
        ),
        "doc-1": StoredPoint(
            vector=np.asarray([0.9, 0.1], dtype=np.float32),
            payload={"tenant_id": "tenant-001", "acl_bucket": 3, "time_bucket": 11},
        ),
    }

    def _create_remote_collection(*, drop_existing: bool) -> None:
        assert drop_existing is True
        adapter._obj = collection

    adapter._create_remote_collection = _create_remote_collection  # type: ignore[method-assign]
    adapter._sync_remote()

    assert len(collection.insert_batches) == 2
    assert collection.insert_batches[0][0]["id"] == "doc-1"
    assert collection.insert_batches[1][0]["id"] == "doc-2"


def test_milvus_sync_remote_rejects_payload_json_overflow() -> None:
    adapter = _make_milvus_adapter()
    collection = _FakeMilvusCollection()
    adapter._records = {
        "doc-1": StoredPoint(
            vector=np.asarray([0.9, 0.1], dtype=np.float32),
            payload={"tenant_id": "tenant-001", "blob": "x" * 70000},
        )
    }

    def _create_remote_collection(*, drop_existing: bool) -> None:
        assert drop_existing is True
        adapter._obj = collection

    adapter._create_remote_collection = _create_remote_collection  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="payload_json max_length=65535"):
        adapter._sync_remote()


def test_milvus_sync_remote_rejects_dimension_mismatch() -> None:
    adapter = _make_milvus_adapter()
    collection = _FakeMilvusCollection()
    adapter._records = {
        "doc-1": StoredPoint(
            vector=np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
            payload={"tenant_id": "tenant-001"},
        )
    }

    def _create_remote_collection(*, drop_existing: bool) -> None:
        assert drop_existing is True
        adapter._obj = collection

    adapter._create_remote_collection = _create_remote_collection  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="vector dimension 3; expected 2"):
        adapter._sync_remote()
