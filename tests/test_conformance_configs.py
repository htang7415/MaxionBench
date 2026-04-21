from __future__ import annotations

import json
from pathlib import Path

from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS


def test_conformance_config_dir_exists_and_has_json_files() -> None:
    config_dir = Path("configs/conformance")
    assert config_dir.exists()
    assert config_dir.is_dir()
    assert any(config_dir.glob("*.json"))


def test_conformance_configs_cover_required_adapters_and_shape() -> None:
    config_dir = Path("configs/conformance")
    files = sorted(config_dir.glob("*.json"))
    adapters: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)

        adapter = str(payload.get("adapter", "")).strip()
        assert adapter
        adapters.add(adapter)

        adapter_options_raw = payload.get("adapter_options_json", None)
        assert isinstance(adapter_options_raw, str)
        adapter_options_payload = json.loads(adapter_options_raw)
        assert isinstance(adapter_options_payload, dict)

        collection = payload.get("collection", None)
        assert isinstance(collection, str) and bool(collection.strip())
        dimension = payload.get("dimension", None)
        assert isinstance(dimension, int) and dimension > 0
        metric = payload.get("metric", None)
        assert isinstance(metric, str) and metric in {"ip", "l2", "cosine"}

    required = set(REQUIRED_ADAPTERS) | {"mock"}
    assert required.issubset(adapters)
