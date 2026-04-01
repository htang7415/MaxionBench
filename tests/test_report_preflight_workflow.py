from __future__ import annotations

from pathlib import Path

import yaml


def test_report_preflight_workflow_validates_before_report() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    assert workflow.exists()

    text = workflow.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert "jobs" in payload
    assert "report_preflight" in payload["jobs"]

    validate_cmd = "maxionbench validate --input artifacts/runs/ci_preflight --strict-schema --json"
    verify_pins_cmd = "maxionbench verify-pins --config-dir configs/scenarios --json"
    verify_pins_paper_cmd = "maxionbench verify-pins --config-dir configs/scenarios_paper --strict-d3-scenario-scale --json"
    verify_dataset_manifests_cmd = "maxionbench verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json"
    verify_behavior_cards_cmd = "maxionbench verify-behavior-cards --behavior-dir docs/behavior --json"
    verify_conformance_configs_cmd = "maxionbench verify-conformance-configs --config-dir configs/conformance --json"
    verify_hygiene_cmd = "pytest -q tests/test_repo_hygiene.py"
    verify_command_docs_cmd = "pytest -q tests/test_command_docs.py"
    verify_workstation_assets_cmd = "pytest -q tests/test_workstation_script.py tests/test_docker_workflow_files.py"
    build_cpu_matrix_cmd = "python -m maxionbench.orchestration.run_matrix"
    build_gpu_matrix_cmd = "--lane gpu"
    snapshot_required_checks_cmd = "maxionbench snapshot-required-checks"
    pre_run_gate_cmd = "maxionbench pre-run-gate --config ci_s1_smoke.yaml --json"
    report_cmd = "maxionbench report"
    inspect_report_output_policy_cmd = "maxionbench inspect-report-output-policy"

    assert verify_pins_cmd in text
    assert verify_pins_paper_cmd in text
    assert verify_dataset_manifests_cmd in text
    assert verify_behavior_cards_cmd in text
    assert verify_conformance_configs_cmd in text
    assert verify_hygiene_cmd in text
    assert verify_command_docs_cmd in text
    assert verify_workstation_assets_cmd in text
    assert build_cpu_matrix_cmd in text
    assert build_gpu_matrix_cmd in text
    assert snapshot_required_checks_cmd in text
    assert pre_run_gate_cmd in text
    assert validate_cmd in text
    assert report_cmd in text
    assert inspect_report_output_policy_cmd in text
    assert "artifacts/ci/workstation_cpu_matrix/**" in text
    assert "artifacts/ci/workstation_gpu_matrix/**" in text
    assert "artifacts/ci/required_checks_snapshot.json" in text
    assert "artifacts/ci/report_output_policy_summary.json" in text
    assert "artifacts/runs/ci_preflight/**" in text
    assert "artifacts/figures/ci_preflight/**" in text

    assert "verify-slurm-plan" not in text
    assert "submit-slurm-plan" not in text
    assert "validate-slurm-snapshots" not in text
    assert "ci-protocol-audit" not in text

    assert text.index(verify_pins_cmd) < text.index(build_cpu_matrix_cmd)
    assert text.index(build_cpu_matrix_cmd) < text.index(validate_cmd)
    assert text.index(validate_cmd) < text.index(report_cmd)
    assert text.index(report_cmd) < text.index(inspect_report_output_policy_cmd)


def test_report_preflight_workflow_has_conformance_readiness_gate_job() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "conformance_readiness_gate" in jobs

    job = jobs["conformance_readiness_gate"]
    assert isinstance(job, dict)
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    runs_blob = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))

    assert "maxionbench conformance-matrix" in runs_blob
    assert "maxionbench verify-conformance-configs --config-dir configs/conformance --json" in runs_blob
    assert "--out-dir artifacts/conformance" in runs_blob
    assert "maxionbench verify-engine-readiness" in runs_blob
    assert "--conformance-matrix artifacts/conformance/conformance_matrix.csv" in runs_blob
    assert "--behavior-dir docs/behavior" in runs_blob
    assert "--allow-gpu-unavailable" in runs_blob
    assert "--allow-nonpass-status" in runs_blob
    assert "--require-mock-pass" in runs_blob
    assert "conformance-readiness-artifacts" in workflow.read_text(encoding="utf-8")


def test_report_preflight_workflow_keeps_required_legacy_jobs() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)

    required_jobs = {
        "report_preflight",
        "conformance_readiness_gate",
        "legacy_migration_path",
        "legacy_resource_profile_path",
        "legacy_ground_truth_metadata_path",
    }
    assert required_jobs.issubset(set(jobs.keys()))
