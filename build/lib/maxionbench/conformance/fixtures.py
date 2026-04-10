"""Shared fixtures for adapter conformance tests."""

from __future__ import annotations

import json
import os
import time
from typing import Iterable

import pytest

from maxionbench.adapters import create_adapter
from maxionbench.conformance.diagnostics import (
    emit_adapter_context,
    emit_post_create_diagnostics,
    emit_pre_create_diagnostics,
)
from maxionbench.schemas.adapter_contract import UpsertRecord


@pytest.fixture()
def adapter() -> Iterable[object]:
    adapter_name = os.environ.get("MAXIONBENCH_CONFORMANCE_ADAPTER", "mock")
    options_json = os.environ.get("MAXIONBENCH_CONFORMANCE_ADAPTER_OPTIONS_JSON", "{}")
    options = json.loads(options_json)
    collection = os.environ.get("MAXIONBENCH_CONFORMANCE_COLLECTION", "conformance")
    dimension = int(os.environ.get("MAXIONBENCH_CONFORMANCE_DIMENSION", "4"))
    metric = os.environ.get("MAXIONBENCH_CONFORMANCE_METRIC", "ip")

    emit_adapter_context(
        adapter_name=adapter_name,
        adapter_options=options,
        collection=collection,
        dimension=dimension,
        metric=metric,
    )
    emit_pre_create_diagnostics(adapter_name=adapter_name, adapter_options=options)
    create_started = time.perf_counter()
    inst = create_adapter(adapter_name, **options)
    create_adapter_latency_s = time.perf_counter() - create_started
    emit_post_create_diagnostics(
        adapter_name=adapter_name,
        adapter=inst,
        collection=collection,
        create_adapter_latency_s=create_adapter_latency_s,
    )
    inst.create(collection=collection, dimension=dimension, metric=metric)
    yield inst
    inst.drop(collection=collection)


@pytest.fixture()
def seed_records() -> list[UpsertRecord]:
    return [
        UpsertRecord(
            id="doc-1",
            vector=[0.9, 0.1, 0.0, 0.0],
            payload={"tenant_id": "tenant-001", "acl_bucket": 1, "time_bucket": 11},
        ),
        UpsertRecord(
            id="doc-2",
            vector=[0.1, 0.9, 0.0, 0.0],
            payload={"tenant_id": "tenant-002", "acl_bucket": 1, "time_bucket": 17},
        ),
        UpsertRecord(
            id="doc-3",
            vector=[0.0, 0.0, 1.0, 0.0],
            payload={"tenant_id": "tenant-001", "acl_bucket": 2, "time_bucket": 23},
        ),
    ]
