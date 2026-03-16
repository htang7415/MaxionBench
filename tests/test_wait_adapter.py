from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

    monkeypatch.setattr(wait_adapter_mod, "create_adapter", _fake_create_adapter)
    summary = wait_adapter_mod.wait_for_adapter(
        adapter_name="qdrant",
        adapter_options={"host": "qdrant", "port": "6333"},
        timeout_s=1.0,
        poll_interval_s=0.0,
    )
    assert captured["name"] == "qdrant"
    assert captured["kwargs"] == {"host": "qdrant", "port": "6333"}
    assert summary["ready"] is True


def test_wait_adapter_main_loads_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "qdrant",
                "engine_version": "server",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D1",
                "dataset_hash": "synthetic-d1-v1",
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
