"""PostgreSQL + pgvector adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Sequence

from maxionbench.schemas.adapter_contract import (
    AdapterStats,
    QueryRequest,
    QueryResult,
    UpsertRecord,
    Vector,
)

from .base import BaseAdapter


@dataclass(frozen=True)
class _PgVectorConfig:
    dsn: str
    schema: str = "public"
    connect_timeout_s: float = 10.0


class PgVectorAdapter(BaseAdapter):
    """pgvector adapter using psycopg SQL execution."""

    def __init__(
        self,
        dsn: str = "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
        schema: str = "public",
        connect_timeout_s: float = 10.0,
        index_method: str | None = None,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "psycopg is required for PgVectorAdapter. Install with `pip install psycopg[binary]`."
            ) from exc
        self._psycopg = psycopg
        self._cfg = _PgVectorConfig(dsn=dsn, schema=schema, connect_timeout_s=connect_timeout_s)
        self._writer = psycopg.connect(dsn, autocommit=False, connect_timeout=int(connect_timeout_s))
        self._reader = psycopg.connect(dsn, autocommit=True, connect_timeout=int(connect_timeout_s))
        self._collection = ""
        self._dimension = 0
        self._metric = "ip"
        self._index_params: dict[str, Any] = {}
        if index_method is not None:
            self._index_params["index_method"] = str(index_method)
        self._search_params: dict[str, Any] = {}
        self._deleted_total = 0
        self._created_at = time.monotonic()

    def create(self, collection: str, dimension: int, metric: str = "ip") -> None:
        self._collection = collection
        self._dimension = dimension
        self._metric = metric
        metric_ops = self._metric_ops(metric)
        vector_type = f"VECTOR({int(dimension)})"
        table_ident = self._table_ident(collection)
        index_ident = self._index_ident(collection)
        with self._writer.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                self._sql(
                    "DROP TABLE IF EXISTS {schema}.{table}",
                    schema=self._cfg.schema,
                    table=table_ident,
                )
            )
            cur.execute(
                self._sql(
                    self._create_table_query(vector_type=vector_type),
                    schema=self._cfg.schema,
                    table=table_ident,
                )
            )
            method = str(self._index_params.get("index_method", "ivfflat")).strip().lower()
            if method not in {"ivfflat", "hnsw", "none", "flat"}:
                method = "ivfflat"
            if method not in {"none", "flat"}:
                with_clause = self._index_with_clause(method)
                cur.execute(
                    self._sql(
                        (
                            "CREATE INDEX {index_name} ON {schema}.{table} "
                            f"USING {method} (embedding {metric_ops}){with_clause}"
                        ),
                        index_name=index_ident,
                        schema=self._cfg.schema,
                        table=table_ident,
                    )
                )
        self._writer.commit()
        self._created_at = time.monotonic()
        self._deleted_total = 0

    def drop(self, collection: str) -> None:
        table_ident = self._table_ident(collection)
        with self._writer.cursor() as cur:
            cur.execute(
                self._sql(
                    "DROP TABLE IF EXISTS {schema}.{table}",
                    schema=self._cfg.schema,
                    table=table_ident,
                )
            )
        self._writer.commit()
        if collection == self._collection:
            self._collection = ""
            self._deleted_total = 0

    def reset(self, collection: str) -> None:
        dimension = self._dimension or 1
        metric = self._metric
        self.drop(collection)
        self.create(collection=collection, dimension=dimension, metric=metric)

    def healthcheck(self) -> bool:
        try:
            with self._reader.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
            return bool(row and int(row[0]) == 1)
        except Exception:
            return False

    def bulk_upsert(self, records: Sequence[UpsertRecord]) -> int:
        if not records:
            return 0
        table_ident = self._table_ident(self._collection)
        values_sql = ", ".join(["(%s, %s::vector, %s::jsonb)"] * len(records))
        stmt = self._sql(
            (
                "INSERT INTO {schema}.{table} (id, embedding, payload) "
                f"VALUES {values_sql} "
                "ON CONFLICT (id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
            ),
            schema=self._cfg.schema,
            table=table_ident,
        )
        params: list[Any] = []
        for record in records:
            params.extend(
                [
                    str(record.id),
                    self._vector_literal(record.vector),
                    json.dumps(dict(record.payload), sort_keys=True),
                ]
            )
        with self._writer.cursor() as cur:
            cur.execute(stmt, params)
        self._writer.commit()
        return len(records)

    def query(self, request: QueryRequest) -> list[QueryResult]:
        table_ident = self._table_ident(self._collection)
        score_sql = self._score_expression()
        filters_sql = ""
        params: list[Any] = [self._vector_literal(request.vector)]
        if request.filters:
            filters_sql = "WHERE payload @> %s::jsonb"
            params.append(json.dumps(dict(request.filters), sort_keys=True))
        params.append(int(request.top_k))

        query_sql = self._sql(
            (
                "SELECT id, payload, "
                f"{score_sql} AS score "
                "FROM {schema}.{table} "
                f"{filters_sql} "
                "ORDER BY score DESC, id ASC "
                "LIMIT %s"
            ),
            schema=self._cfg.schema,
            table=table_ident,
        )
        with self._reader.cursor() as cur:
            self._apply_search_params(cur)
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        return [QueryResult(id=str(row[0]), score=float(row[2]), payload=dict(row[1] or {})) for row in rows]

    def batch_query(self, requests: Sequence[QueryRequest]) -> list[list[QueryResult]]:
        return [self.query(request) for request in requests]

    def insert(self, record: UpsertRecord) -> None:
        self.bulk_upsert([record])

    def update_vectors(self, ids: Sequence[str], vectors: Sequence[Vector]) -> int:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        table_ident = self._table_ident(self._collection)
        stmt = self._sql(
            "UPDATE {schema}.{table} SET embedding = %s::vector WHERE id = %s",
            schema=self._cfg.schema,
            table=table_ident,
        )
        updated = 0
        with self._writer.cursor() as cur:
            for doc_id, vector in zip(ids, vectors):
                cur.execute(stmt, (self._vector_literal(vector), str(doc_id)))
                updated += max(cur.rowcount, 0)
        self._writer.commit()
        return updated

    def update_payload(self, ids: Sequence[str], payload: Mapping[str, Any]) -> int:
        table_ident = self._table_ident(self._collection)
        stmt = self._sql(
            "UPDATE {schema}.{table} SET payload = payload || %s::jsonb WHERE id = %s",
            schema=self._cfg.schema,
            table=table_ident,
        )
        payload_json = json.dumps(dict(payload), sort_keys=True)
        updated = 0
        with self._writer.cursor() as cur:
            for doc_id in ids:
                cur.execute(stmt, (payload_json, str(doc_id)))
                updated += max(cur.rowcount, 0)
        self._writer.commit()
        return updated

    def delete(self, ids: Sequence[str]) -> int:
        table_ident = self._table_ident(self._collection)
        stmt = self._sql(
            "DELETE FROM {schema}.{table} WHERE id = %s",
            schema=self._cfg.schema,
            table=table_ident,
        )
        deleted = 0
        with self._writer.cursor() as cur:
            for doc_id in ids:
                cur.execute(stmt, (str(doc_id),))
                deleted += max(cur.rowcount, 0)
        self._writer.commit()
        self._deleted_total += deleted
        return deleted

    def flush_or_commit(self) -> None:
        self._writer.commit()

    def set_index_params(self, params: Mapping[str, Any]) -> None:
        self._index_params = dict(params)

    def set_search_params(self, params: Mapping[str, Any]) -> None:
        self._search_params = dict(params)

    def optimize_or_compact(self) -> None:
        self.flush_or_commit()
        with self._reader.cursor() as cur:
            cur.execute(
                self._sql(
                    "VACUUM ANALYZE {schema}.{table}",
                    schema=self._cfg.schema,
                    table=self._table_ident(self._collection),
                )
            )

    def stats(self) -> AdapterStats:
        if not self._collection:
            return AdapterStats(
                vector_count=0,
                deleted_count=0,
                index_size_bytes=0,
                ram_usage_bytes=0,
                disk_usage_bytes=0,
                engine_uptime_s=time.monotonic() - self._created_at,
            )
        with self._reader.cursor() as cur:
            cur.execute(self._sql("SELECT COUNT(*) FROM {schema}.{table}", schema=self._cfg.schema, table=self._collection))
            count_row = cur.fetchone()
            vector_count = int(count_row[0]) if count_row else 0

            cur.execute("SELECT pg_total_relation_size(to_regclass(%s))", (self._qualified_table(),))
            disk_row = cur.fetchone()
            disk_size = int(disk_row[0] or 0) if disk_row else 0

            cur.execute("SELECT pg_indexes_size(to_regclass(%s))", (self._qualified_table(),))
            index_row = cur.fetchone()
            index_size = int(index_row[0] or 0) if index_row else 0

            cur.execute(
                "SELECT COALESCE(n_dead_tup, 0) FROM pg_stat_user_tables WHERE schemaname = %s AND relname = %s",
                (self._cfg.schema, self._collection),
            )
            dead_row = cur.fetchone()
            dead_tup = int(dead_row[0] or 0) if dead_row else 0

        return AdapterStats(
            vector_count=vector_count,
            deleted_count=max(self._deleted_total, dead_tup),
            index_size_bytes=index_size,
            ram_usage_bytes=0,
            disk_usage_bytes=disk_size,
            engine_uptime_s=time.monotonic() - self._created_at,
        )

    def _apply_search_params(self, cur: Any) -> None:
        probes = self._search_params.get("ivfflat_probes")
        if probes is not None:
            cur.execute("SELECT set_config('ivfflat.probes', %s, false)", (str(max(1, int(probes))),))
        ef_search = self._search_params.get("hnsw_ef_search")
        if ef_search is not None:
            cur.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(max(1, int(ef_search))),))

    def _score_expression(self) -> str:
        metric = self._metric.strip().lower()
        if metric in {"ip", "inner_product", "dot"}:
            return "-(embedding <#> %s::vector)"
        if metric in {"l2", "euclid", "euclidean"}:
            return "-(embedding <-> %s::vector)"
        if metric in {"cos", "cosine"}:
            return "1 - (embedding <=> %s::vector)"
        raise ValueError(f"Unsupported metric for pgvector: {self._metric}")

    @staticmethod
    def _metric_ops(metric: str) -> str:
        normalized = metric.strip().lower()
        if normalized in {"ip", "inner_product", "dot"}:
            return "vector_ip_ops"
        if normalized in {"l2", "euclid", "euclidean"}:
            return "vector_l2_ops"
        if normalized in {"cos", "cosine"}:
            return "vector_cosine_ops"
        raise ValueError(f"Unsupported metric for pgvector: {metric}")

    def _index_with_clause(self, method: str) -> str:
        options: list[str] = []
        if method == "ivfflat":
            lists = self._index_params.get("lists", self._index_params.get("nlist"))
            if lists is not None:
                options.append(f"lists = {max(1, int(lists))}")
        elif method == "hnsw":
            m = self._index_params.get("m", self._index_params.get("M"))
            ef_construction = self._index_params.get(
                "ef_construction",
                self._index_params.get("efConstruction"),
            )
            if m is not None:
                options.append(f"m = {max(1, int(m))}")
            if ef_construction is not None:
                options.append(f"ef_construction = {max(1, int(ef_construction))}")
        if not options:
            return ""
        return " WITH (" + ", ".join(options) + ")"

    def _sql(self, query: str, **identifiers: str) -> Any:
        sql = self._psycopg.sql
        parts = {
            key: sql.Identifier(value)
            for key, value in identifiers.items()
        }
        return sql.SQL(query).format(**parts)

    def _table_ident(self, collection: str) -> str:
        if not collection:
            raise ValueError("collection must be set before issuing operations")
        return collection

    @staticmethod
    def _index_ident(collection: str) -> str:
        return f"{collection}_embedding_idx"

    @classmethod
    def _create_table_query(cls, *, vector_type: str) -> str:
        return (
            "CREATE TABLE {schema}.{table} ("
            "id TEXT PRIMARY KEY, "
            f"embedding {vector_type} NOT NULL, "
            f"payload JSONB NOT NULL DEFAULT {cls._empty_jsonb_literal()})"
        )

    @staticmethod
    def _empty_jsonb_literal() -> str:
        # psycopg.sql.SQL.format() interprets bare `{}` as placeholders.
        return "'{{}}'::jsonb"

    def _qualified_table(self) -> str:
        return f"{self._cfg.schema}.{self._collection}"

    @staticmethod
    def _vector_literal(vector: Vector) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"

    def __del__(self) -> None:
        for conn in (getattr(self, "_writer", None), getattr(self, "_reader", None)):
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
