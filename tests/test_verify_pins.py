from __future__ import annotations

import json
from pathlib import Path

import yaml

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_pins import verify_scenario_config_dir


def test_verify_pins_passes_for_portable_scenario_configs() -> None:
    summary = verify_scenario_config_dir(Path("configs/scenarios_portable"))
    assert summary["pass"] is True
    assert int(summary["files_checked"]) == 3
    assert int(summary["error_count"]) == 0


def test_verify_pins_requires_hotpot_bundle_for_s3(tmp_path: Path) -> None:
    src = Path("configs/scenarios_portable/s3_multi_hop.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["dataset_bundle"] = "FRAMES_PORTABLE"

    out = tmp_path / "s3_multi_hop.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("dataset_bundle" in msg for msg in messages)


def test_verify_pins_detects_portable_clients_grid_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios_portable/s1_single_hop.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["clients_grid"] = [1, 8]

    out = tmp_path / "s1_single_hop.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("clients_grid" in msg for msg in messages)


def test_verify_pins_detects_portable_rpc_baseline_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios_portable/s2_streaming_memory.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["rpc_baseline_requests"] = 5

    out = tmp_path / "s2_streaming_memory.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("rpc_baseline_requests" in msg for msg in messages)


def test_verify_pins_detects_portable_d4_doc_cap_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios_portable/s1_single_hop.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["d4_max_docs"] = 200000

    out = tmp_path / "s1_single_hop.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("d4_max_docs" in msg for msg in messages)


def test_verify_pins_cli_reports_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = Path("configs/scenarios_portable/s3_multi_hop.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["sla_threshold_ms"] = 500.0

    out = tmp_path / "s3_multi_hop.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    code = cli_main(["verify-pins", "--config-dir", str(tmp_path), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1
