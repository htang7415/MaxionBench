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


def test_verify_pins_passes_for_paper_scenario_configs_with_strict_d3_scale() -> None:
    summary = verify_scenario_config_dir(
        Path("configs/scenarios_paper"),
        strict_d3_scenario_scale=True,
    )
    assert summary["pass"] is True
    assert int(summary["files_checked"]) >= 1
    assert int(summary["error_count"]) == 0
    assert summary["strict_d3_scenario_scale"] is True


def test_verify_pins_passes_for_portable_scenario_configs() -> None:
    summary = verify_scenario_config_dir(Path("configs/scenarios_portable"))
    assert summary["pass"] is True
    assert int(summary["files_checked"]) == 3
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


def test_verify_pins_detects_s5_require_hf_backend_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s5_rerank.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["s5_require_hf_backend"] = False

    out = tmp_path / "s5_rerank.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("s5_require_hf_backend" in msg for msg in messages)


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


def test_verify_pins_detects_d3_vector_dim_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s2_filtered_ann.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["vector_dim"] = 64

    out = tmp_path / "s2_filtered_ann.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("vector_dim" in msg for msg in messages)


def test_verify_pins_detects_calibrate_d3_scale_drift(tmp_path: Path) -> None:
    src = Path("configs/scenarios/calibrate_d3.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 10000

    out = tmp_path / "calibrate_d3.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("D3-10M+ scale" in msg for msg in messages)


def test_verify_pins_allows_dev_calibrate_d3_scale_relaxation(tmp_path: Path) -> None:
    src = Path("configs/scenarios/calibrate_d3.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 10000

    out = tmp_path / "calibrate_d3.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path, allow_dev_calibrate_d3_scale=True)
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["allow_dev_calibrate_d3_scale"] is True


def test_verify_pins_strict_d3_scenario_scale_flags_repo_d3_defaults() -> None:
    summary = verify_scenario_config_dir(Path("configs/scenarios"), strict_d3_scenario_scale=True)
    assert summary["pass"] is False
    assert summary["strict_d3_scenario_scale"] is True
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("D3 scenarios must run at D3-10M+ scale in strict mode" in msg for msg in messages)


def test_verify_pins_strict_d3_scenario_scale_passes_for_10m_d3_config(tmp_path: Path) -> None:
    src = Path("configs/scenarios/s2_filtered_ann.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 10_000_000
    payload["d3_k_clusters"] = 4096

    out = tmp_path / "s2_filtered_ann.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path, strict_d3_scenario_scale=True)
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["strict_d3_scenario_scale"] is True


def test_verify_pins_cli_allows_dev_calibrate_d3_scale_relaxation(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    src = Path("configs/scenarios/calibrate_d3.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 10000

    out = tmp_path / "calibrate_d3.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    code = cli_main(
        [
            "verify-pins",
            "--config-dir",
            str(tmp_path),
            "--allow-dev-calibrate-d3-scale",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is True
    assert parsed["allow_dev_calibrate_d3_scale"] is True


def test_verify_pins_cli_strict_d3_scenario_scale_reports_failure(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    src = Path("configs/scenarios/s2_filtered_ann.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["num_vectors"] = 5000

    out = tmp_path / "s2_filtered_ann.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    code = cli_main(
        [
            "verify-pins",
            "--config-dir",
            str(tmp_path),
            "--strict-d3-scenario-scale",
            "--json",
        ]
    )
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["strict_d3_scenario_scale"] is True


def test_verify_pins_detects_missing_paper_calibration_real_data_requirement(tmp_path: Path) -> None:
    src = Path("configs/scenarios_paper/calibrate_d3.yaml")
    payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["calibration_require_real_data"] = False

    out = tmp_path / "calibrate_d3.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_scenario_config_dir(tmp_path, strict_d3_scenario_scale=True)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("calibration_require_real_data" in msg for msg in messages)
