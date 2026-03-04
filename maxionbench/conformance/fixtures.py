"""Shared fixtures for adapter conformance tests."""

from __future__ import annotations

import json
import os
from typing import Iterable

import pytest

from maxionbench.adapters import create_adapter
from maxionbench.schemas.adapter_contract import UpsertRecord


@pytest.fixture()
def adapter() -> Iterable[object]:
    adapter_name = os.environ.get("MAXIONBENCH_CONFORMANCE_ADAPTER", "mock")
    options_json = os.environ.get("MAXIONBENCH_CONFORMANCE_ADAPTER_OPTIONS_JSON", "{}")
    options = json.loads(options_json)
    collection = os.environ.get("MAXIONBENCH_CONFORMANCE_COLLECTION", "conformance")
    dimension = int(os.environ.get("MAXIONBENCH_CONFORMANCE_DIMENSION", "4"))
    metric = os.environ.get("MAXIONBENCH_CONFORMANCE_METRIC", "ip")

    inst = create_adapter(adapter_name, **options)
    inst.create(collection=collection, dimension=dimension, metric=metric)
    yield inst
    inst.drop(collection=collection)


@pytest.fixture()
def seed_records() -> list[UpsertRecord]:
    return [
        UpsertRecord(id="doc-1", vector=[0.9, 0.1, 0.0, 0.0], payload={"tenant": "t1", "acl": "a"}),
        UpsertRecord(id="doc-2", vector=[0.1, 0.9, 0.0, 0.0], payload={"tenant": "t2", "acl": "a"}),
        UpsertRecord(id="doc-3", vector=[0.0, 0.0, 1.0, 0.0], payload={"tenant": "t1", "acl": "b"}),
    ]
