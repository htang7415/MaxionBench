from __future__ import annotations

import sys
import types

from maxionbench.adapters.faiss_cpu import FaissCpuAdapter
from maxionbench.schemas.adapter_contract import QueryRequest


class _FakeIndex:
    def search(self, query, top_k):  # type: ignore[no-untyped-def]
        del query, top_k
        return [[0.9, 0.1, 0.0]], [[0, 2, -1]]


def test_faiss_query_ignores_stale_upper_bound_indices(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace())
    adapter = FaissCpuAdapter()
    adapter.create(collection="c", dimension=2, metric="ip")
    adapter._records = {"doc-1": adapter._pending_upserts.get("doc-1") or types.SimpleNamespace(payload={"x": 1}, vector=None)}
    adapter._id_by_pos = ["doc-1"]
    adapter._index = _FakeIndex()

    results = adapter.query(QueryRequest(vector=[1.0, 0.0], top_k=3))

    assert [item.id for item in results] == ["doc-1"]
