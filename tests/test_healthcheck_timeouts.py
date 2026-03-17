from __future__ import annotations

from typing import Any

from maxionbench.adapters.opensearch import OpenSearchAdapter
from maxionbench.adapters.weaviate import WeaviateAdapter


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.content = b"{}"

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return dict(self._body)


def test_opensearch_healthcheck_timeout_is_separate_from_normal_requests(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    timeouts: list[float] = []

    def _fake_request(*, timeout: float, **kwargs: Any) -> _FakeResponse:
        timeouts.append(float(timeout))
        return _FakeResponse()

    monkeypatch.setattr("maxionbench.adapters.opensearch.requests.request", _fake_request)
    adapter = OpenSearchAdapter(timeout_s=30.0, healthcheck_timeout_s=5.0)
    assert adapter.healthcheck() is True
    adapter.drop("collection")
    assert timeouts == [5.0, 30.0]


def test_weaviate_healthcheck_timeout_is_separate_from_normal_requests(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    timeouts: list[float] = []

    def _fake_request(*, timeout: float, **kwargs: Any) -> _FakeResponse:
        timeouts.append(float(timeout))
        return _FakeResponse()

    monkeypatch.setattr("maxionbench.adapters.weaviate.requests.request", _fake_request)
    adapter = WeaviateAdapter(timeout_s=30.0, healthcheck_timeout_s=5.0)
    assert adapter.healthcheck() is True
    adapter.drop("collection")
    assert timeouts == [5.0, 30.0]
