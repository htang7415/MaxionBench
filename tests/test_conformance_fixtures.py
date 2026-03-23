from __future__ import annotations

from maxionbench.conformance.fixtures import seed_records


def test_conformance_seed_records_use_remote_adapter_filter_fields() -> None:
    records = seed_records.__wrapped__()  # type: ignore[attr-defined]
    assert len(records) == 3
    for record in records:
        assert {"tenant_id", "acl_bucket", "time_bucket"} <= set(record.payload.keys())

