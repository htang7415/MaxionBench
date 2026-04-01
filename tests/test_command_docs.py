from __future__ import annotations

from pathlib import Path


def test_command_md_is_linux_workstation_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Linux Workstation Commands" in text
    assert "`install -> download -> preprocess -> calibrate -> workstation matrix`" in text
    assert 'python -m pip install -e ".[dev,engines,reporting,datasets]"' in text
    assert "python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json" in text
    assert "python -m maxionbench.cli run --config configs/scenarios_paper/calibrate_d3.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "bash run_workstation.sh --lane cpu" in text
    assert "bash run_workstation.sh --lane gpu" in text
    assert "bash run_workstation.sh --lane all" in text
    assert "--scratch-dir /mnt/nvme/maxionbench" in text
    assert "bash run_docker_scenario.sh --config configs/scenarios/s1_ann_frontier_qdrant.yaml" in text
    assert "--benchmark-service benchmark-gpu" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final" in text

    assert "run_slurm_pipeline.sh" not in text
    assert "submit-slurm-plan" not in text
    assert ".env.slurm" not in text
    assert "Euler" not in text
    assert "NREL" not in text
    assert "Apptainer" not in text


def test_command_mac_md_is_reduced_mac_doc() -> None:
    text = Path("command-mac.md").read_text(encoding="utf-8")

    assert "# MaxionBench Mac Mini M4 Commands" in text
    assert "This is the reduced Mac lane:" in text
    assert "Do not treat it as the full paper D2/D3 or CUDA-backed GPU path." in text
    assert 'python -m pip install -e ".[dev,engines,reporting,datasets]"' in text
    assert "python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json" in text
    assert "bash preprocess_all_datasets.sh" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/calibrate_d3.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s1_ann_frontier_qdrant_local.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli run --config configs/scenarios/s6_fusion.yaml --seed 42 --repeats 1 --no-retry" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final" in text

    assert "run_workstation.sh" not in text
    assert "run_slurm_pipeline.sh" not in text
    assert "submit-slurm-plan" not in text
