from __future__ import annotations

import sys
import types

from maxionbench.adapters.pgvector import PgVectorAdapter
from maxionbench.schemas.adapter_contract import UpsertRecord


class _FakeCursor:
    def __init__(self) -> None:
        self.rowcount = 1

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def execute(self, stmt, params=None) -> None:  # type: ignore[no-untyped-def]
        del stmt, params
        self.rowcount = 1

    def fetchone(self):  # type: ignore[no-untyped-def]
        return (1,)

    def fetchall(self):  # type: ignore[no-untyped-def]
        return []


class _FakeConnection:
    def __init__(self) -> None:
        self.commit_count = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    def commit(self) -> None:
        self.commit_count += 1


class _FakeSQLTemplate:
    def __init__(self, query: str) -> None:
        self._query = query

    def format(self, **identifiers: str) -> str:
        return self._query.format(**identifiers)


class _FakeSQLModule:
    @staticmethod
    def Identifier(value: str) -> str:
        return value

    @staticmethod
    def SQL(query: str) -> _FakeSQLTemplate:
        return _FakeSQLTemplate(query)


def test_pgvector_write_paths_commit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    connections: list[_FakeConnection] = []

    def _connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        conn = _FakeConnection()
        connections.append(conn)
        return conn

    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=_connect, sql=_FakeSQLModule()))
    adapter = PgVectorAdapter()
    writer = connections[0]
    adapter._collection = "items"
    adapter._dimension = 2

    adapter.bulk_upsert([UpsertRecord(id="doc-1", vector=[1.0, 0.0], payload={})])
    adapter.update_vectors(ids=["doc-1"], vectors=[[0.0, 1.0]])
    adapter.update_payload(ids=["doc-1"], payload={"tenant": "a"})
    adapter.delete(ids=["doc-1"])

    assert writer.commit_count == 4
