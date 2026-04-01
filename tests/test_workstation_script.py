from __future__ import annotations

from pathlib import Path
import subprocess


def test_workstation_script_exists_and_references_local_matrix_workflow() -> None:
    script = Path("run_workstation.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert 'SCENARIO_CONFIG_DIR="configs/scenarios_paper"' in text
    assert 'ENGINE_CONFIG_DIR="configs/engines"' in text
    assert 'LANE="cpu"' in text
    assert "--lane <cpu|gpu|all>" in text
    assert 'python -m maxionbench.orchestration.run_matrix' in text
    assert 'python -m maxionbench.orchestration.local_preflight' in text
    assert 'bash run_docker_scenario.sh' in text
    assert 'benchmark-gpu' in text
    assert 'MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI' in text
    assert 'artifacts/workstation_runs' in text
    assert 'processed_dataset_path' in text
    assert 'missing_real_dataset_file' in text
    assert 'RUN_MATRIX_DIR' in text
    assert 'RUN_PREFLIGHT_DIR' in text
    assert 'Figure helper script:' in text
    assert 'run_slurm_pipeline.sh' not in text
    assert 'submit-slurm-plan' not in text
    assert 'verify-slurm-plan' not in text
    assert 'ci-protocol-audit' not in text


def test_workstation_script_is_bash_parseable() -> None:
    completed = subprocess.run(
        ["bash", "-n", "run_workstation.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
