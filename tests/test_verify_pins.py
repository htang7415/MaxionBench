from __future__ import annotations

import json
from pathlib import Path

import yaml

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_pins import verify_scenario_config_dir


def test_verify_pins_passes_for_repo_scenario_configs() -> None:
    summary = verify_scenario_config_dir(Path("configs/scenarios"))
    assert summary["pass"] is True
    assert int(summary["files_checked"]) >= 1
    assert int(summary["error_count"]) == 0


def test_verify_pins_detects_drift_in_temp_config_dir(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s5_rerank.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["s5_reranker_revision_tag"] = "2026-03-05"

    out = tmp_path / "s5_rerank.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("s5_reranker_revision_tag" in msg for msg in messages)


def test_verify_pins_cli_reports_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = Path("configs/scenarios/s4_hybrid.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["rrf_k"] = 59

    out = tmp_path / "s4_hybrid.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    code = cli_main(["verify-pins", "--config-dir", str(tmp_path), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1


def test_verify_pins_detects_rhu_weight_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s1_ann_frontier.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["w_c"] = 0.2

    out = tmp_path / "s1_ann_frontier.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("w_c" in msg for msg in messages)


def test_verify_pins_detects_missing_required_crag_path_for_d4_real(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s4_hybrid_d4_real.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["d4_crag_path"] = None

    out = tmp_path / "s4_hybrid_d4_real.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("d4_crag_path" in msg for msg in messages)


def test_verify_pins_detects_required_crag_source_drift_for_d4_real(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s4_hybrid_d4_real.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["d4_crag_source"] = "other-org/CRAG"

    out = tmp_path / "s4_hybrid_d4_real.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("d4_crag_source" in msg for msg in messages)


def test_verify_pins_detects_s4_clients_grid_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s4_hybrid.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["clients_grid"] = [8, 16]

    out = tmp_path / "s4_hybrid.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("clients_grid" in msg for msg in messages)


def test_verify_pins_allows_d3_50m_k8192_tier(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s2_filtered_ann.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 50_000_000
    payload["d3_k_clusters"] = 8192

    out = tmp_path / "s2_filtered_ann.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_verify_pins_detects_d3_50m_k_cluster_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s2_filtered_ann.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 50_000_000
    payload["d3_k_clusters"] = 4096

    out = tmp_path / "s2_filtered_ann.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("d3_k_clusters" in msg for msg in messages)
