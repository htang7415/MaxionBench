"""LanceDB in-process adapter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from ._exact import StoredPoint, matches_filter, normalize_metric
from .base import BaseAdapter


class LanceDbInprocAdapter(BaseAdapter):
    """LanceDB in-process adapter with explicit flush semantics."""

    def __init__(self, uri: str | None = None) -> None:
        try:
            import lancedb  # type: ignore
        except ImportError as exc:
            raise ImportError("lancedb is required for LanceDbInprocAdapter. Install with `pip install lancedb`.") from exc
        self._lancedb = lancedb
        resolved_uri = uri or str((Path(tempfile.gettempdir()) / "maxionbench" / "lancedb" / "inproc").resolve())
        self._uri = str(Path(resolved_uri).resolve())
        Path(self._uri).mkdir(parents=True, exist_ok=True)
        _check_filesystem_supports_rename(self._uri)
        self._db = self._lancedb.connect(self._uri)
        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._created_at = time.monotonic()
        self._records: dict[str, StoredPoint] = {}
        self._pending_upserts: dict[str, StoredPoint] = {}
        self._pending_deletes: set[str] = set()
        self._deleted_total = 0
        self._index_params: dict[str, Any] = {}
        self._search_params: dict[str, Any] = {}
        self._table: Any | None = None

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._dimension = int(dimension)
        self._metric = normalize_metric(metric)
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._table = None
        self._drop_table_if_exists()
        self._created_at = time.monotonic()

    def drop(self, collection: str) -> None:
        if collection != self._collection and collection:
            try:
                self._db.drop_table(collection)
            except Exception:
                pass
            return
        self._drop_table_if_exists()
        self._collection = ""
        self._dimension = 0
        self._records.clear()
        self._pending_upserts.clear()
        self._pending_deletes.clear()
        self._deleted_total = 0
        self._table = None

    def reset(self, collection: str) -> None:
        dimension = self._dimension or 1
        metric = self._metric
        self.drop(collection)
        self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        return bool(self._collection)

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        for record in records:
            key = str(record.id)
            self._pending_upserts[key] = StoredPoint(vector=self._to_vector(record.vector), payload=dict(record.payload))
            self._pending_deletes.discard(key)
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        if self._table is None:
            if not self._records:
                return []
            raise RuntimeError("LanceDB table unavailable for committed records; flush_or_commit did not materialize the table")
        query_vec = self._to_vector(request.vector)
        search_limit = request.top_k if not request.filters else max(request.top_k, len(self._records))
        search = self._table.search(query_vec.tolist()).limit(int(max(search_limit, 1)))
        frame = search.to_pandas()
        results: list[QueryResult] = []
        for rank, row in enumerate(frame.to_dict(orient="records")):
            doc_id = str(row.get("id") or "")
            payload_raw = row.get("payload")
            payload = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}
            if not matches_filter(payload, request.filters):
                continue
            results.append(
                QueryResult(
                    id=doc_id,
                    score=_lance_row_score(row=row, rank=rank),
                    payload=payload,
                )
            )
            if len(results) >= request.top_k:
                break
        return results

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests]

    def insert(self, record: UpsertRecord) -> None:
        self.bulk_upsert([record])

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have same length")
        updated = 0
        for doc_id, vector in zip(ids, vectors):
            key = str(doc_id)
            base = self._pending_upserts.get(key) or self._records.get(key)
            if base is None:
                continue
            self._pending_upserts[key] = StoredPoint(vector=self._to_vector(vector), payload=dict(base.payload))
            self._pending_deletes.discard(key)
            updated += 1
        return updated

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        updated = 0
        for doc_id in ids:
            key = str(doc_id)
            base = self._pending_upserts.get(key) or self._records.get(key)
            if base is None:
                continue
            merged = dict(base.payload)
            merged.update(payload)
            self._pending_upserts[key] = StoredPoint(vector=base.vector.copy(), payload=merged)
            self._pending_deletes.discard(key)
            updated += 1
        return updated

    def delete(self, ids: Sequence[str]) -> int:
        for doc_id in ids:
            key = str(doc_id)
            self._pending_deletes.add(key)
            self._pending_upserts.pop(key, None)
        return len(ids)

    def flush_or_commit(self) -> None:
        for doc_id in sorted(self._pending_deletes):
            if doc_id in self._records:
                self._deleted_total += 1
            self._records.pop(doc_id, None)
        for doc_id in sorted(self._pending_upserts):
            self._records[doc_id] = self._pending_upserts[doc_id]
        self._pending_deletes.clear()
        self._pending_upserts.clear()
        self._sync_table()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        # LanceDB compaction/optimize APIs vary by version; flush ensures durable state here.
        self.flush_or_commit()

    def stats(self) -> AdapterStats:
        vector_count = len(self._records)
        disk_size = self._disk_usage_bytes()
        ram_size = vector_count * self._dimension * 4
        return AdapterStats(
            vector_count=vector_count,
            deleted_count=self._deleted_total,
            index_size_bytes=disk_size,
            ram_usage_bytes=ram_size,
            disk_usage_bytes=disk_size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _sync_table(self) -> None:
        self._drop_table_if_exists()
        if not self._records:
            self._table = None
            return
        rows = [
            {
                "id": doc_id,
                "vector": point.vector.tolist(),
                "payload": dict(point.payload),
            }
            for doc_id, point in sorted(self._records.items(), key=lambda item: item[0])
        ]
        self._table = self._db.create_table(self._collection, data=rows)

    def _drop_table_if_exists(self) -> None:
        if not self._collection:
            return
        try:
            self._db.drop_table(self._collection)
        except Exception:
            pass

    def _to_vector(self, vector: Vector) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        if self._dimension and arr.shape[0] != self._dimension:
            raise ValueError(f"vector dimension mismatch: expected {self._dimension}, got {arr.shape[0]}")
        return arr

    def _disk_usage_bytes(self) -> int:
        root = Path(self._uri)
        if not root.exists():
            return 0
        return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))


def _check_filesystem_supports_rename(uri: str) -> None:
    """Raise RuntimeError early if the filesystem at uri does not support atomic rename.

    LanceDB requires rename() for commit atomicity.  Some external or network-mounted
    filesystems (e.g. AFP/SMB volumes on macOS) return ENOTSUP (errno 45), which causes
    a cryptic deep failure inside Lance.  Better to catch it at adapter construction time.
    """
    import os

    probe_dir = Path(uri)
    src = probe_dir / ".maxionbench_rename_probe"
    dst = probe_dir / ".maxionbench_rename_probe_dst"
    try:
        src.write_bytes(b"")
        os.rename(src, dst)
        dst.unlink(missing_ok=True)
    except OSError as exc:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"LanceDB storage path {uri!r} is on a filesystem that does not support "
            f"atomic rename (errno {exc.errno}: {exc.strerror}). "
            "Move MAXIONBENCH_LANCEDB_INPROC_URI to a local filesystem (e.g. /tmp)."
        ) from exc


def _lance_row_score(*, row: Mapping[str, Any], rank: int) -> float:
    for key in ("_score", "score"):
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    for key in ("_distance", "distance"):
        value = row.get(key)
        try:
            return -float(value)
        except (TypeError, ValueError):
            continue
    return float(-rank)
