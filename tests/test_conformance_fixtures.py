from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from maxionbench.conformance import diagnostics as diagnostics_mod
from maxionbench.conformance.fixtures import seed_records


def test_conformance_seed_records_use_remote_adapter_filter_fields() -> None:
    records = seed_records.__wrapped__()  # type: ignore[attr-defined]
    assert len(records) == 3
    for record in records:
        assert {"tenant_id", "acl_bucket", "time_bucket"} <= set(record.payload.keys())


def test_emit_pre_create_diagnostics_uses_hyphenated_faiss_gpu_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        diagnostics_mod,
        "_emit",
        lambda payload: emitted.append(dict(payload)),
    )
    monkeypatch.setattr(
        diagnostics_mod,
        "_faiss_gpu_diagnostics",
        lambda **kwargs: {"adapter": "faiss-gpu", "event": "conformance_adapter_diagnostics"},
    )

    diagnostics_mod.emit_pre_create_diagnostics(adapter_name="faiss-gpu", adapter_options={"device_id": 0})

    assert emitted == [{"adapter": "faiss-gpu", "event": "conformance_adapter_diagnostics"}]


def test_faiss_gpu_diagnostics_reports_hyphenated_adapter_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_faiss = SimpleNamespace(
        __version__="1.0-test",
        get_num_gpus=lambda: 1,
        StandardGpuResources=lambda: object(),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    payload = diagnostics_mod._faiss_gpu_diagnostics(adapter_options={"device_id": 2})

    assert payload["adapter"] == "faiss-gpu"
    assert payload["faiss_import_ok"] is True
    assert payload["device_id"] == 2
    assert payload["cuda_device_count"] == 1
    assert payload["standard_gpu_resources_ok"] is True
