from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxionbench.adapters import faiss_gpu as faiss_gpu_mod
from maxionbench.adapters.pgvector import PgVectorAdapter


def test_pgvector_create_table_query_escapes_json_default_braces() -> None:
    query = PgVectorAdapter._create_table_query(vector_type="VECTOR(4)")

    assert "'{{}}'::jsonb" in query
    assert "'{}'::jsonb" not in query


def test_faiss_gpu_adapter_raises_clean_runtime_error_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_faiss = SimpleNamespace(
        __version__="1.14.1-test",
        StandardGpuResources=lambda: object(),
    )

    def _fake_parent_init(self) -> None:
        self._faiss = fake_faiss

    monkeypatch.setattr(faiss_gpu_mod.FaissCpuAdapter, "__init__", _fake_parent_init)
    monkeypatch.setattr(
        faiss_gpu_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="CUDA error 209 no kernel image is available for execution on the device",
        ),
    )
    faiss_gpu_mod.FaissGpuAdapter._runtime_probe_cache.clear()

    with pytest.raises(RuntimeError, match="no kernel image is available for execution on the device"):
        faiss_gpu_mod.FaissGpuAdapter(device_id=0)
