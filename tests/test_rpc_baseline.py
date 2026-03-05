from __future__ import annotations

import pytest

from maxionbench.runtime.rpc_baseline import minimal_rpc_request_fn


class _DummyAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int] | str] = []

    def healthcheck(self) -> bool:
        self.calls.append("healthcheck")
        return True

    def query(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(("query", int(request.top_k), len(request.vector)))
        return []


def test_minimal_rpc_request_fn_uses_healthcheck_and_query() -> None:
    adapter = _DummyAdapter()
    request_fn = minimal_rpc_request_fn(adapter=adapter, vector_dim=6)
    request_fn()
    assert adapter.calls == ["healthcheck", ("query", 1, 6)]


def test_minimal_rpc_request_fn_rejects_invalid_vector_dim() -> None:
    adapter = _DummyAdapter()
    with pytest.raises(ValueError, match="vector_dim"):
        minimal_rpc_request_fn(adapter=adapter, vector_dim=0)
