"""Adapter conformance tests required before benchmarking engines."""

from __future__ import annotations

import os

from maxionbench.schemas.adapter_contract import QueryRequest, UpsertRecord

from .fixtures import adapter, seed_records  # noqa: F401 (pytest fixture discovery)


def test_healthcheck(adapter) -> None:
    assert adapter.healthcheck() is True


def test_flush_controls_visibility(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    before = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=3))
    if _expect_deferred_visibility():
        assert before == []

    adapter.flush_or_commit()
    after = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=3))
    assert [item.id for item in after][:1] == ["doc-1"]


def test_filter_correctness(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()
    filtered = adapter.query(
        QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=10, filters={"tenant_id": "tenant-001"})
    )
    assert {item.id for item in filtered} == {"doc-1", "doc-3"}


def test_query_empty_collection_returns_no_results(adapter) -> None:
    results = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=5))
    assert results == []


def test_empty_bulk_upsert_is_noop(adapter) -> None:
    assert adapter.bulk_upsert([]) == 0
    adapter.flush_or_commit()
    assert adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)) == []


def test_update_vectors_semantics(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()

    adapter.update_vectors(ids=["doc-2"], vectors=[[1.0, 0.0, 0.0, 0.0]])
    before = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=1))
    if _expect_deferred_visibility():
        assert before[0].id == "doc-1"

    adapter.flush_or_commit()
    after = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=1))
    assert after[0].id == "doc-2"


def test_update_payload_semantics(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()

    adapter.update_payload(ids=["doc-3"], payload={"acl_bucket": 1})
    adapter.flush_or_commit()
    filtered = adapter.query(
        QueryRequest(vector=[0.0, 0.0, 1.0, 0.0], top_k=10, filters={"acl_bucket": 1})
    )
    assert {item.id for item in filtered} == {"doc-1", "doc-2", "doc-3"}


def test_delete_semantics(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()

    adapter.delete(ids=["doc-1"])
    adapter.flush_or_commit()
    results = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=10))
    assert "doc-1" not in {item.id for item in results}


def test_double_delete_is_stable(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()

    adapter.delete(ids=["doc-1"])
    adapter.flush_or_commit()
    adapter.delete(ids=["doc-1"])
    adapter.flush_or_commit()

    results = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=10))
    assert "doc-1" not in {item.id for item in results}


def test_insert_and_batch_query(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.insert(
        UpsertRecord(
            id="doc-4",
            vector=[0.0, 1.0, 0.0, 0.0],
            payload={"tenant_id": "tenant-003", "acl_bucket": 3, "time_bucket": 29},
        )
    )

    before = adapter.query(QueryRequest(vector=[0.0, 1.0, 0.0, 0.0], top_k=10))
    if _expect_deferred_visibility():
        assert "doc-4" not in {item.id for item in before}

    adapter.flush_or_commit()
    batch = adapter.batch_query(
        [
            QueryRequest(vector=[0.0, 1.0, 0.0, 0.0], top_k=2),
            QueryRequest(vector=[0.0, 0.0, 1.0, 0.0], top_k=2),
        ]
    )
    assert len(batch) == 2
    assert batch[0][0].id in {"doc-2", "doc-4"}
    assert batch[1][0].id == "doc-3"


def test_stats_minimum_fields(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()
    stats = adapter.stats()
    assert stats.vector_count == 3
    assert stats.deleted_count >= 0
    assert stats.index_size_bytes >= 0
    assert stats.ram_usage_bytes >= 0
    assert stats.disk_usage_bytes >= 0
    assert stats.engine_uptime_s >= 0


def test_optimize_or_compact_does_not_break_read_path(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()
    adapter.optimize_or_compact()
    results = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=1))
    assert results


def test_repeated_flush_is_stable(adapter, seed_records) -> None:
    adapter.bulk_upsert(seed_records)
    adapter.flush_or_commit()
    first = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=3))
    adapter.flush_or_commit()
    second = adapter.query(QueryRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=3))
    assert [item.id for item in first] == [item.id for item in second]


def _expect_deferred_visibility() -> bool:
    adapter_name = os.environ.get("MAXIONBENCH_CONFORMANCE_ADAPTER", "mock").strip().lower()
    if adapter_name == "mock":
        return True
    return False
