from __future__ import annotations

import json
from pathlib import Path

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_conformance_configs import verify_conformance_config_dir


def test_verify_conformance_configs_passes_for_repo_catalog() -> None:
    summary = verify_conformance_config_dir(config_dir=Path("configs/conformance"), allow_gpu_unavailable=True)
    assert summary["pass"] is True
    assert int(summary["files_checked"]) >= 1
    assert summary["missing_required_adapters"] == []
    assert summary["allow_gpu_unavailable"] is True


def test_verify_conformance_configs_flags_missing_dir(tmp_path: Path) -> None:
    summary = verify_conformance_config_dir(config_dir=tmp_path / "does_not_exist")
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("does not exist" in msg for msg in messages)


def test_verify_conformance_configs_detects_invalid_adapter_options_json(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "mock.json").write_text(
        json.dumps(
            {
                "adapter": "mock",
                "adapter_options_json": "{not-json",
                "collection": "conformance",
                "dimension": 4,
                "metric": "ip",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = verify_conformance_config_dir(config_dir=config_dir)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("adapter_options_json is not valid JSON" in msg for msg in messages)


def test_verify_conformance_configs_flags_missing_required_adapter(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg_missing_qdrant"
    config_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path("configs/conformance")
    for path in sorted(src_dir.glob("*.json")):
        if path.name == "qdrant_local.json":
            continue
        (config_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = verify_conformance_config_dir(config_dir=config_dir, allow_gpu_unavailable=True)
    assert summary["pass"] is False
    assert "qdrant" in summary["missing_required_adapters"]


def test_verify_conformance_configs_cli_failure_json(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    code = cli_main(
        [
            "verify-conformance-configs",
            "--config-dir",
            str(tmp_path / "missing_cfg"),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is False
