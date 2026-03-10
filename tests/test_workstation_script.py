from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_workstation_script_exists_and_references_paper_lane_checks() -> None:
    script = Path("run_workstation.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "maxionbench verify-pins --config-dir \"${SCENARIO_CONFIG_DIR}\" --strict-d3-scenario-scale --json" in text
    assert "run_submit_slurm_plan() {" in text
    assert "--scenario-config-dir \"${SCENARIO_CONFIG_DIR}\"" in text
    assert "--output-root \"${RUN_RESULTS_SLURM}\"" in text
    assert "--skip-gpu \\" in text
    assert "--dry-run \\" in text
    assert "--json | tee \"${SLURM_SUBMIT_PAPER_SKIP_GPU_DRY_RUN_JSON}\"" in text
    assert "maxionbench ci-protocol-audit" in text
    assert "--config-dir configs/scenarios \\" in text
    assert "--config-dir \"${SCENARIO_CONFIG_DIR}\"" in text
    assert "CI_PROTOCOL_AUDIT_DEFAULT_JSON" in text
    assert "CI_PROTOCOL_AUDIT_PAPER_JSON" in text
    assert "--strict-d3-scenario-scale" in text
    assert "--slurm-profile <name>" in text
    assert "--skip-s6" in text
    assert "--prefetch-datasets" in text
    assert "cmd+=(\"${SLURM_PROFILE_ARGS[@]}\")" in text
    assert "cmd+=(\"${SLURM_S6_ARGS[@]}\")" in text
    assert "cmd+=(\"${SLURM_PREFETCH_ARGS[@]}\")" in text
    assert "artifacts/workstation_runs" in text
    assert "RUN_BUNDLE_ROOT" in text
    assert "RUN_RESULTS_SLURM" in text
    assert "RUN_FIGURES_MILESTONES" in text
    assert "RUN_FIGURES_FINAL" in text
    assert "render_figures.sh" in text
    assert "Run report saved:" in text
    assert "trap finalize_report EXIT" in text
    assert "--launch" in text
    assert "--container-runtime <name>" in text
    assert "--container-image <path>" in text
    assert "--container-bind <spec>" in text
    assert "--hf-cache-dir <path>" in text
    assert "cmd+=(\"${SLURM_CONTAINER_ARGS[@]}\")" in text
    assert "profiles_local.example.yaml" in text
    assert "profiles_local.yaml" in text
    assert "MAXIONBENCH_D3_DATASET_PATH" in text
    assert "MAXIONBENCH_D3_DATASET_SHA256" in text
    assert "requires real D3 vectors" in text
    assert "dataset_prefetch" in text


def test_workstation_script_dry_run_handles_empty_optional_submit_arrays(tmp_path: Path) -> None:
    source_script = Path("run_workstation.sh")
    script_path = tmp_path / "run_workstation.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    (tmp_path / "configs" / "scenarios_paper").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "scenarios_paper" / "calibrate_d3.yaml").write_text("{}\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_calls.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MAXIONBENCH_STUB_LOG}"
printf '{"pass": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)

    completed = subprocess.run(
        [
            "bash",
            "run_workstation.sh",
            "--skip-pytest",
            "--skip-calibration",
            "--container-runtime",
            "apptainer",
            "--container-image",
            "/shared/containers/maxionbench.sif",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = stub_log.read_text(encoding="utf-8").splitlines()
    submit_calls = [line for line in calls if line.startswith("submit-slurm-plan ")]
    assert len(submit_calls) == 3
    for line in submit_calls:
        assert "--container-runtime apptainer" in line
        assert "--container-image /shared/containers/maxionbench.sif" in line
        assert "--output-root artifacts/workstation_runs/" in line
        assert "--dry-run" in line
    assert all("--slurm-profile" not in line for line in submit_calls)
    assert all("--skip-s6" not in line for line in submit_calls)
    assert all("--prefetch-datasets" not in line for line in submit_calls)


def test_command_md_mentions_workstation_script() -> None:
    text = Path("command.md").read_text(encoding="utf-8")
    assert "./run_workstation.sh" in text
    assert "./run_workstation.sh --launch" in text
    assert "./run_workstation.sh --prefetch-datasets" in text
    assert "./run_workstation.sh --skip-s6 --launch" in text
    assert "./run_workstation.sh --launch --cpu-only" in text
    assert "MAXIONBENCH_D3_DATASET_PATH=/abs/path/to/laion_d3_vectors.npy ./run_workstation.sh" in text
    assert "--container-runtime apptainer --container-image /shared/containers/maxionbench.sif" in text
    assert "artifacts/workstation_runs/<run_id>/" in text
    assert "helpers/render_figures.sh" in text
    assert "--output-root artifacts/workstation_runs/<run_id>/results/slurm" in text
