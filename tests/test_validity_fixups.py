from __future__ import annotations

from pathlib import Path
import time

import yaml

from maxionbench.adapters.lancedb_inproc import LanceDbInprocAdapter
from maxionbench.adapters.pgvector import PgVectorAdapter
from maxionbench.schemas.adapter_contract import UpsertRecord


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        return None

    def execute(self, stmt, params=None) -> None:  # type: ignore[no-untyped-def]
        self.calls.append((stmt, params))


class _FakeWriter:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.commit_calls = 0

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_calls += 1


def test_pgvector_bulk_upsert_batches_into_one_insert() -> None:
    adapter = object.__new__(PgVectorAdapter)
    adapter._cfg = type("Cfg", (), {"schema": "public"})()
    adapter._collection = "bench"
    adapter._writer = _FakeWriter(_FakeCursor())
    adapter._sql = lambda query, **identifiers: query  # type: ignore[assignment]
    adapter._table_ident = lambda collection: collection  # type: ignore[assignment]

    records = [
        UpsertRecord(id="a", vector=[1.0, 0.0], payload={"tenant_id": "tenant-001"}),
        UpsertRecord(id="b", vector=[0.0, 1.0], payload={"tenant_id": "tenant-002"}),
    ]
    count = adapter.bulk_upsert(records)
    cursor = adapter._writer._cursor
    assert count == 2
    assert len(cursor.calls) == 1
    stmt, params = cursor.calls[0]
    assert "ON CONFLICT (id) DO UPDATE" in str(stmt)
    assert str(stmt).count("(%s, %s::vector, %s::jsonb)") == 2
    assert len(params) == 6


def test_pgvector_create_skips_index_when_index_method_none() -> None:
    cursor = _FakeCursor()
    writer = _FakeWriter(cursor)
    adapter = object.__new__(PgVectorAdapter)
    adapter._cfg = type("Cfg", (), {"schema": "public"})()
    adapter._writer = writer
    adapter._index_params = {"index_method": "none"}
    adapter._deleted_total = 0
    adapter._created_at = 0.0

    def _fake_sql(query: str, **identifiers: str) -> str:
        rendered = query
        for key, value in identifiers.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered

    adapter._sql = _fake_sql  # type: ignore[assignment]
    adapter._table_ident = lambda collection: collection  # type: ignore[assignment]
    adapter._index_ident = lambda collection: f"{collection}_embedding_idx"  # type: ignore[assignment]
    adapter.create(collection="bench", dimension=4, metric="ip")

    statements = [str(stmt) for stmt, _ in cursor.calls]
    assert any("CREATE TABLE public.bench" in stmt for stmt in statements)
    assert all("CREATE INDEX" not in stmt for stmt in statements)
    assert writer.commit_calls == 1


def test_lancedb_stats_use_measured_disk_bytes(tmp_path: Path) -> None:
    data_file = tmp_path / "table" / "segment.bin"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_bytes(b"1234567")

    adapter = object.__new__(LanceDbInprocAdapter)
    adapter._uri = str(tmp_path)
    adapter._records = {"doc-1": object()}
    adapter._dimension = 4
    adapter._deleted_total = 0
    adapter._created_at = time.monotonic()

    stats = adapter.stats()
    assert stats.disk_usage_bytes == 7
    assert stats.index_size_bytes == 7


def test_paper_scenario_configs_require_strict_timing_mode() -> None:
    root = Path("configs/scenarios_paper")
    for path in sorted(root.glob("*.yaml")):
        if path.name == "calibrate_d3.yaml":
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("phase_timing_mode") == "strict"


def test_remote_gpu_profile_uses_placeholder_account() -> None:
    payload = yaml.safe_load(Path("maxionbench/orchestration/slurm/profiles_local.yaml").read_text(encoding="utf-8"))
    assert payload["remote_gpu"]["base"][-1] == ["--account", "<cluster-account>"]
