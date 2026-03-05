from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.orchestration.slurm.preflight import evaluate_preflight
from maxionbench.runtime.ports import allocate_named_ports, allocate_port_range


def test_allocate_port_range_and_named_ports() -> None:
    ports = allocate_port_range(count=3, base=25000, span=1000, offset=10)
    assert len(ports) == 3
    assert ports[1] == ports[0] + 1

    named = allocate_named_ports(["a", "b", "a"], base=26000, span=1000)
    assert set(named.keys()) == {"a", "b"}
    assert named["b"] == named["a"] + 1


def test_preflight_uses_manifest_when_available(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "D3"
    assert summary["dataset_bytes"] > 0
    assert summary["fallback_config"] == "configs/scenarios/s2_filtered_ann.yaml"


def test_preflight_estimate_without_manifest(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_unknown.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "UNKNOWN"
    assert summary["dataset_bytes"] > 0
    assert summary["engine_bytes"] > 0
    assert summary["temp_bytes"] > 0


def test_slurm_common_runs_pre_run_gate_before_runner() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_PRE_RUN_GATE" in text
    assert "MAXIONBENCH_ALLOW_GPU_UNAVAILABLE" in text
    assert "MAXIONBENCH_CONFORMANCE_MATRIX" in text
    gate_marker = "pre-run-gate"
    runner_marker = "python -m maxionbench.orchestration.runner"
    assert gate_marker in text
    assert runner_marker in text
    assert text.index(gate_marker) < text.index(runner_marker)
