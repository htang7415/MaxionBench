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
    assert "--skip-completed" in text
    assert "--continue-on-failure" in text
    assert "--resume-bundle <path|id>" in text
    assert "--engine-filter <csv>" in text
    assert "--template-filter <csv>" in text
    assert "--no-prebuild" in text
    assert "--gpu-benchmark-mode <mode>" in text
    assert 'python -m maxionbench.orchestration.run_matrix' in text
    assert 'python -m maxionbench.orchestration.local_preflight' in text
    assert 'bash run_docker_scenario.sh' in text
    assert 'benchmark-gpu' in text
    assert 'GPU_BENCHMARK_MODE="docker"' in text
    assert 'D3_PARAMS_PATH="artifacts/calibration/d3_params.yaml"' in text
    assert 'mb() {' in text
    assert 'scenario_requires_d3_params' in text
    assert 'selected strict D3 rows require ${D3_PARAMS_PATH}' in text
    assert '--d3-params "${D3_PARAMS_PATH}"' in text
    assert '--local-benchmark' in text
    assert 'export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${RUN_RESULTS_LOCAL}/lancedb/service"' in text
    assert 'if [[ "${group}" == "gpu" && "${GPU_BENCHMARK_MODE}" == "local" ]]; then' in text
    assert 'if [[ "${group}" == "gpu" ]] || service_engine "${engine}"; then' in text
    assert 'docker_has_nvidia_runtime' in text
    assert 'docker daemon does not expose an NVIDIA runtime' in text
    assert 'nvidia-container-toolkit' in text
    assert 'docker_gpu_runtime_ready' in text
    assert 'benchmark-gpu cannot access a GPU through Docker on this workstation' in text
    assert 'local_gpu_runtime_ready' in text
    assert 'faiss import does not expose GPU bindings' in text
    assert 'local gpu benchmark mode is not ready in the current Python environment' in text
    assert 'MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI' in text
    assert 'artifacts/workstation_runs' in text
    assert 'run_status_is_success' in text
    assert 'processed_dataset_path' in text
    assert 'expand_env_placeholders' in text
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
