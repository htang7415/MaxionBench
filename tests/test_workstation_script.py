from __future__ import annotations

from pathlib import Path


def test_workstation_script_exists_and_references_paper_lane_checks() -> None:
    script = Path("run_workstation.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "maxionbench verify-pins --config-dir \"${SCENARIO_CONFIG_DIR}\" --strict-d3-scenario-scale --json" in text
    assert "maxionbench submit-slurm-plan \\" in text
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
    assert "\"${SLURM_PROFILE_ARGS[@]}\"" in text
    assert "\"${SLURM_S6_ARGS[@]}\"" in text
    assert "artifacts/workstation_runs" in text
    assert "RUN_BUNDLE_ROOT" in text
    assert "RUN_RESULTS_SLURM" in text
    assert "RUN_FIGURES_MILESTONES" in text
    assert "RUN_FIGURES_FINAL" in text
    assert "render_figures.sh" in text
    assert "Run report saved:" in text
    assert "trap finalize_report EXIT" in text
    assert "--launch" in text


def test_command_md_mentions_workstation_script() -> None:
    text = Path("command.md").read_text(encoding="utf-8")
    assert "./run_workstation.sh" in text
    assert "./run_workstation.sh --launch" in text
    assert "./run_workstation.sh --skip-s6 --launch" in text
    assert "./run_workstation.sh --launch --cpu-only" in text
    assert "artifacts/workstation_runs/<run_id>/" in text
    assert "helpers/render_figures.sh" in text
    assert "--output-root artifacts/workstation_runs/<run_id>/results/slurm" in text
