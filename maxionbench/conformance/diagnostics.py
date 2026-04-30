"""Adapter-specific diagnostics emitted alongside conformance stdout artifacts."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping
import uuid


def emit_adapter_context(
    *,
    adapter_name: str,
    adapter_options: Mapping[str, Any],
    collection: str,
    dimension: int,
    metric: str,
) -> None:
    _emit(
        {
            "event": "conformance_adapter_context",
            "adapter": adapter_name,
            "adapter_options": dict(adapter_options),
            "collection": collection,
            "dimension": int(dimension),
            "metric": metric,
        }
    )


def emit_pre_create_diagnostics(*, adapter_name: str, adapter_options: Mapping[str, Any]) -> None:
    del adapter_name, adapter_options


def emit_post_create_diagnostics(
    *,
    adapter_name: str,
    adapter: Any,
    collection: str,
    create_adapter_latency_s: float,
) -> None:
    if adapter_name == "pgvector":
        _emit(
            _pgvector_diagnostics(
                adapter=adapter,
                collection=collection,
                create_adapter_latency_s=create_adapter_latency_s,
            )
        )


def _pgvector_diagnostics(*, adapter: Any, collection: str, create_adapter_latency_s: float) -> dict[str, Any]:
    diag_table = f"{collection}_diag_{uuid.uuid4().hex[:8]}"
    payload: dict[str, Any] = {
        "event": "conformance_adapter_diagnostics",
        "adapter": "pgvector",
        "connect_latency_s": round(float(create_adapter_latency_s), 6),
        "schema": str(getattr(getattr(adapter, "_cfg", None), "schema", "public")),
        "diag_table": diag_table,
    }
    writer = getattr(adapter, "_writer", None)
    if writer is None:
        payload["connectivity_ok"] = False
        payload["error"] = "pgvector adapter has no writer connection"
        return payload

    schema = str(getattr(getattr(adapter, "_cfg", None), "schema", "public"))
    create_extension_started = time.perf_counter()
    with writer.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    writer.commit()
    payload["create_extension_latency_s"] = round(time.perf_counter() - create_extension_started, 6)

    with writer.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    payload["extension_version"] = str(row[0]) if row and row[0] is not None else None

    create_stmt = adapter._sql("CREATE TABLE {schema}.{table} (id INTEGER PRIMARY KEY)", schema=schema, table=diag_table)
    drop_stmt = adapter._sql("DROP TABLE {schema}.{table}", schema=schema, table=diag_table)

    create_started = time.perf_counter()
    with writer.cursor() as cur:
        cur.execute(create_stmt)
    writer.commit()
    payload["create_table_latency_s"] = round(time.perf_counter() - create_started, 6)

    drop_started = time.perf_counter()
    with writer.cursor() as cur:
        cur.execute(drop_stmt)
    writer.commit()
    payload["drop_table_latency_s"] = round(time.perf_counter() - drop_started, 6)
    payload["connectivity_ok"] = True
    return payload


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _exc_summary(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__
