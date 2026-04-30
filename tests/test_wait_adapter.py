from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maxionbench.runtime import healthcheck as healthcheck_mod
from maxionbench.tools import wait_adapter as wait_adapter_mod


class _FakeAdapter:
    def __init__(self, healthy_after: int = 1) -> None:
        self._remaining = healthy_after

    def healthcheck(self) -> bool:
        self._remaining -= 1
        return self._remaining <= 0


def test_wait_for_adapter_polls_until_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_adapter(name: str, **kwargs: object) -> _FakeAdapter:
        captured["name"] = name
        captured["kwargs"] = dict(kwargs)
        return _FakeAdapter(healthy_after=2)

    def _fake_wait_for_port_ready(host: str, port: int, timeout_s: float, poll_interval_s: float, connect_timeout_s: float = 1.0) -> None:
        captured["tcp_probe"] = {
            "host": host,
            "port": port,
            "timeout_s": timeout_s,
            "poll_interval_s": poll_interval_s,
            "connect_timeout_s": connect_timeout_s,
        }

    monkeypatch.setattr(wait_adapter_mod, "create_adapter", _fake_create_adapter)
    monkeypatch.setattr(wait_adapter_mod, "wait_for_port_ready", _fake_wait_for_port_ready)
    summary = wait_adapter_mod.wait_for_adapter(
        adapter_name="qdrant",
        adapter_options={"host": "qdrant", "port": "6333"},
        timeout_s=1.0,
        poll_interval_s=0.0,
    )
    assert captured["name"] == "qdrant"
    assert captured["kwargs"] == {"host": "qdrant", "port": "6333"}
    assert captured["tcp_probe"] == {
        "host": "qdrant",
        "port": 6333,
        "timeout_s": 1.0,
        "poll_interval_s": 0.0,
        "connect_timeout_s": 1.0,
    }
    assert summary["tcp_probe"] == {"host": "qdrant", "port": 6333}
    assert summary["ready"] is True


def test_wait_adapter_main_loads_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "qdrant",
                "engine_version": "server",
                "scenario": "s1_single_hop",
                "dataset_bundle": "D4",
                "dataset_hash": "portable-d4-v1",
                "no_retry": True,
                "adapter_options": {
                    "host": "${MAXIONBENCH_QDRANT_HOST:-127.0.0.1}",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXIONBENCH_QDRANT_HOST", "qdrant")

    captured: dict[str, object] = {}

    def _fake_wait_for_adapter(*, adapter_name: str, adapter_options: dict[str, object], timeout_s: float, poll_interval_s: float) -> dict[str, object]:
        captured["adapter_name"] = adapter_name
        captured["adapter_options"] = dict(adapter_options)
        captured["timeout_s"] = timeout_s
        captured["poll_interval_s"] = poll_interval_s
        return {"ready": True}

    monkeypatch.setattr(wait_adapter_mod, "wait_for_adapter", _fake_wait_for_adapter)
    code = wait_adapter_mod.main(["--config", str(cfg_path), "--timeout-s", "30", "--poll-interval-s", "2"])
    assert code == 0
    assert captured["adapter_name"] == "qdrant"
    assert captured["adapter_options"] == {"host": "qdrant"}
    assert captured["timeout_s"] == 30.0
    assert captured["poll_interval_s"] == 2.0


def test_wait_for_port_ready_succeeds_for_listening_socket() -> None:
    class _FakeSocket:
        def __enter__(self) -> "_FakeSocket":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
            return False

    original = healthcheck_mod.socket.create_connection
    calls: list[tuple[tuple[str, int], float]] = []

    def _fake_create_connection(address: tuple[str, int], timeout: float = 1.0) -> _FakeSocket:
        calls.append((address, timeout))
        return _FakeSocket()

    healthcheck_mod.socket.create_connection = _fake_create_connection  # type: ignore[assignment]
    try:
        healthcheck_mod.wait_for_port_ready("127.0.0.1", 9000, timeout_s=0.2, poll_interval_s=0.01, connect_timeout_s=0.05)
    finally:
        healthcheck_mod.socket.create_connection = original  # type: ignore[assignment]
    assert calls == [(("127.0.0.1", 9000), 0.05)]


def test_wait_for_port_ready_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def _always_fail(*args: object, **kwargs: object) -> object:
        raise OSError("unreachable")

    monkeypatch.setattr(healthcheck_mod.socket, "create_connection", _always_fail)
    with pytest.raises(TimeoutError, match="did not become reachable"):
        healthcheck_mod.wait_for_port_ready("127.0.0.1", 9999, timeout_s=0.02, poll_interval_s=0.0, connect_timeout_s=0.01)
