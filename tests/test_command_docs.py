from __future__ import annotations

from pathlib import Path


def test_command_md_is_portable_agentic_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Portable-Agentic Commands" in text
    assert "`install -> conformance -> download -> one-time preprocess -> run B0/B1/B2 -> report -> archive`" in text
    assert 'pip install -e ".[dev,engines,reporting,datasets]"' in text
    assert "maxionbench conformance-matrix --config-dir configs/conformance --out-dir artifacts/conformance --timeout-s 30" in text
    assert "--datasets scifact,fiqa,crag,frames" in text
    assert "maxionbench preprocess-frames-portable" in text
    assert "bash run_workstation.sh --profile portable-agentic --budget b0" in text
    assert "bash run_workstation.sh --profile portable-agentic --budget b1" in text
    assert "bash run_workstation.sh --profile portable-agentic --budget b2" in text
    assert "configs/scenarios_portable/s1_single_hop.yaml" in text
    assert "configs/scenarios_portable/s2_streaming_memory.yaml" in text
    assert "configs/scenarios_portable/s3_multi_hop.yaml" in text
    assert "bash save_results_bundle.sh --profile portable-agentic" in text

    assert "run_docker_scenario.sh" not in text
    assert "--gpu-benchmark-mode local" not in text
    assert "calibrate_d3" not in text
