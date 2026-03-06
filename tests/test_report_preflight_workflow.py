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
    report_job = payload["jobs"]["report_preflight"]
    assert isinstance(report_job, dict)
    steps = report_job.get("steps", [])
    assert isinstance(steps, list)

    validate_cmd = "maxionbench validate --input artifacts/runs/ci_preflight --strict-schema --json"
    verify_pins_cmd = "maxionbench verify-pins --config-dir configs/scenarios --json"
    verify_dataset_manifests_cmd = (
        "maxionbench verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json"
    )
    verify_behavior_cards_cmd = "maxionbench verify-behavior-cards --behavior-dir docs/behavior --json"
    verify_hygiene_cmd = "pytest -q tests/test_repo_hygiene.py"
    verify_command_docs_cmd = "pytest -q tests/test_command_docs.py"
    verify_figure_policy_sync_cmd = "pytest -q tests/test_report_figure_policy_sync.py"
    verify_migration_docs_cmd = "pytest -q tests/test_migration_docs.py"
    verify_branch_policy_sync_cmd = "pytest -q tests/test_branch_protection_policy_sync.py"
    verify_slurm_plan_cmd = "maxionbench verify-slurm-plan --json"
    verify_slurm_plan_skip_gpu_cmd = "maxionbench verify-slurm-plan --skip-gpu --json"
    verify_slurm_submit_plan_cmd = "maxionbench submit-slurm-plan --dry-run --json"
    verify_slurm_submit_plan_skip_gpu_cmd = "maxionbench submit-slurm-plan --skip-gpu --dry-run --json"
    validate_slurm_snapshot_step_name = "Validate Slurm plan snapshot payloads"
    validate_slurm_snapshot_cmd = "maxionbench validate-slurm-snapshots"
    snapshot_required_checks_cmd = "maxionbench snapshot-required-checks"
    snapshot_required_checks_strict = "--strict"
    pre_run_gate_cmd = "maxionbench pre-run-gate --config ci_s1_smoke.yaml --json"
    report_cmd = "maxionbench report"
    inspect_report_output_policy_cmd = "maxionbench inspect-report-output-policy"
    inspect_report_output_policy_input = "--input artifacts/figures/ci_preflight"
    inspect_report_output_policy_output = "--output artifacts/ci/report_output_policy_summary.json"
    inspect_report_output_policy_strict = "--strict"
    inspect_report_output_policy_json = "--json"
    ci_protocol_audit_cmd = "maxionbench ci-protocol-audit"
    ci_protocol_audit_manifest_dir = "--manifest-dir maxionbench/datasets/manifests"
    ci_protocol_audit_output = "--output artifacts/ci/ci_protocol_audit.json"
    ci_protocol_audit_require_report_policy = "--require-report-policy"
    assert verify_pins_cmd in text
    assert verify_dataset_manifests_cmd in text
    assert verify_behavior_cards_cmd in text
    assert verify_hygiene_cmd in text
    assert verify_command_docs_cmd in text
    assert verify_figure_policy_sync_cmd in text
    assert verify_migration_docs_cmd in text
    assert verify_branch_policy_sync_cmd in text
    assert verify_slurm_plan_cmd in text
    assert verify_slurm_plan_skip_gpu_cmd in text
    assert verify_slurm_submit_plan_cmd in text
    assert verify_slurm_submit_plan_skip_gpu_cmd in text
    assert validate_slurm_snapshot_step_name in text
    assert validate_slurm_snapshot_cmd in text
    assert snapshot_required_checks_cmd in text
    assert snapshot_required_checks_strict in text
    assert pre_run_gate_cmd in text
    assert validate_cmd in text
    assert report_cmd in text
    assert inspect_report_output_policy_cmd in text
    assert inspect_report_output_policy_input in text
    assert inspect_report_output_policy_output in text
    assert inspect_report_output_policy_strict in text
    assert inspect_report_output_policy_json in text
    assert ci_protocol_audit_cmd in text
    assert ci_protocol_audit_manifest_dir in text
    assert ci_protocol_audit_output in text
    assert ci_protocol_audit_require_report_policy in text
    inspect_steps = [
        step
        for step in steps
        if isinstance(step, dict) and inspect_report_output_policy_cmd in str(step.get("run", ""))
    ]
    assert len(inspect_steps) == 1
    inspect_run = str(inspect_steps[0].get("run", ""))
    assert inspect_report_output_policy_input in inspect_run
    assert inspect_report_output_policy_output in inspect_run
    assert inspect_report_output_policy_strict in inspect_run
    assert inspect_report_output_policy_json in inspect_run
    assert text.index(verify_pins_cmd) < text.index(validate_cmd)
    assert text.index(verify_dataset_manifests_cmd) < text.index(validate_cmd)
    assert text.index(verify_behavior_cards_cmd) < text.index(validate_cmd)
    assert text.index(verify_hygiene_cmd) < text.index(validate_cmd)
    assert text.index(verify_command_docs_cmd) < text.index(validate_cmd)
    assert text.index(verify_figure_policy_sync_cmd) < text.index(validate_cmd)
    assert text.index(verify_migration_docs_cmd) < text.index(validate_cmd)
    assert text.index(verify_branch_policy_sync_cmd) < text.index(validate_cmd)
    assert text.index(verify_slurm_plan_cmd) < text.index(validate_cmd)
    assert text.index(verify_slurm_plan_skip_gpu_cmd) < text.index(validate_cmd)
    assert text.index(verify_slurm_submit_plan_cmd) < text.index(validate_cmd)
    assert text.index(verify_slurm_submit_plan_skip_gpu_cmd) < text.index(validate_cmd)
    assert text.index(validate_slurm_snapshot_step_name) < text.index(validate_cmd)
    assert text.index(snapshot_required_checks_cmd) < text.index(validate_cmd)
    assert text.index(pre_run_gate_cmd) < text.index(validate_cmd)
    assert text.index(validate_cmd) < text.index(report_cmd)
    assert text.index(report_cmd) < text.index(inspect_report_output_policy_cmd)
    assert text.index(inspect_report_output_policy_cmd) < text.index(ci_protocol_audit_cmd)

    assert "artifacts/ci/required_checks_snapshot.json" in text
    assert "artifacts/ci/report_output_policy_summary.json" in text
    assert "artifacts/ci/slurm_plan_verify.json" in text
    assert "artifacts/ci/slurm_plan_verify_skip_gpu.json" in text
    assert "artifacts/ci/slurm_submit_plan_dry_run.json" in text
    assert "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json" in text
    assert "artifacts/ci/slurm_snapshot_validation.json" in text
    assert "artifacts/ci/ci_protocol_audit.json" in text
    assert "artifacts/runs/ci_preflight/**" in text
    assert "artifacts/figures/ci_preflight/**" in text

    inspect_idx = next(
        idx
        for idx, step in enumerate(steps)
        if isinstance(step, dict) and inspect_report_output_policy_cmd in str(step.get("run", ""))
    )
    upload_steps = [
        (idx, step)
        for idx, step in enumerate(steps)
        if isinstance(step, dict) and step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert len(upload_steps) == 1
    upload_idx, upload_step = upload_steps[0]
    assert inspect_idx < upload_idx
    assert upload_step.get("if") == "always()"
    upload_with = upload_step.get("with", {})
    assert isinstance(upload_with, dict)
    upload_paths = str(upload_with.get("path", ""))
    assert "artifacts/ci/required_checks_snapshot.json" in upload_paths
    assert "artifacts/ci/report_output_policy_summary.json" in upload_paths
    assert "artifacts/ci/slurm_plan_verify.json" in upload_paths
    assert "artifacts/ci/slurm_plan_verify_skip_gpu.json" in upload_paths
    assert "artifacts/ci/slurm_submit_plan_dry_run.json" in upload_paths
    assert "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json" in upload_paths
    assert "artifacts/ci/slurm_snapshot_validation.json" in upload_paths
    assert "artifacts/ci/ci_protocol_audit.json" in upload_paths


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
    assert "--config-dir configs/conformance" in runs_blob
    assert "--out-dir artifacts/conformance" in runs_blob
    assert "maxionbench verify-engine-readiness" in runs_blob
    assert "--conformance-matrix artifacts/conformance/conformance_matrix.csv" in runs_blob
    assert "--behavior-dir docs/behavior" in runs_blob
    assert "--allow-gpu-unavailable" in runs_blob
    assert "--allow-nonpass-status" in runs_blob
    assert "--require-mock-pass" in runs_blob
    assert "--json" in runs_blob
    text = workflow.read_text(encoding="utf-8")
    assert "conformance-readiness-artifacts" in text
    assert "artifacts/conformance/**" in text


def test_report_preflight_workflow_has_legacy_migration_path() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "legacy_migration_path" in jobs

    legacy_job = jobs["legacy_migration_path"]
    assert isinstance(legacy_job, dict)
    steps = legacy_job.get("steps", [])
    assert isinstance(steps, list)
    step_names = [str(step.get("name", "")) for step in steps if isinstance(step, dict)]

    assert "Assert report fails before migration" in step_names
    assert "Backfill stage timing fields" in step_names
    assert "Validate migrated artifacts" in step_names
    assert "Generate report after migration" in step_names
    assert "Upload legacy migration artifacts" in step_names

    assert step_names.index("Assert report fails before migration") < step_names.index("Backfill stage timing fields")
    assert step_names.index("Backfill stage timing fields") < step_names.index("Generate report after migration")

    runs_blob = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))
    assert "maxionbench migrate-stage-timing --input artifacts/runs/ci_legacy" in runs_blob
    assert "maxionbench validate --input artifacts/runs/ci_legacy --strict-schema --json" in runs_blob
    assert "grep -q \"migrate-stage-timing\"" in runs_blob
    assert "artifacts/runs/ci_legacy/**" in workflow.read_text(encoding="utf-8")
    assert "artifacts/figures/ci_legacy/**" in workflow.read_text(encoding="utf-8")


def test_report_preflight_workflow_has_legacy_resource_profile_path() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "legacy_resource_profile_path" in jobs

    legacy_job = jobs["legacy_resource_profile_path"]
    assert isinstance(legacy_job, dict)
    steps = legacy_job.get("steps", [])
    assert isinstance(steps, list)
    step_names = [str(step.get("name", "")) for step in steps if isinstance(step, dict)]

    assert "Drop RHU resource fields to emulate legacy artifact" in step_names
    assert "Assert strict validation fails for legacy resource artifact" in step_names
    assert "Capture legacy-compatible validation warnings" in step_names
    assert "Assert report fails before regeneration" in step_names
    assert "Upload legacy resource artifacts" in step_names

    assert step_names.index("Drop RHU resource fields to emulate legacy artifact") < step_names.index(
        "Assert strict validation fails for legacy resource artifact"
    )
    assert step_names.index("Capture legacy-compatible validation warnings") < step_names.index(
        "Assert report fails before regeneration"
    )

    runs_blob = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))
    assert "maxionbench validate --input artifacts/runs/ci_legacy_resource --strict-schema --json" in runs_blob
    assert "maxionbench validate --input artifacts/runs/ci_legacy_resource --legacy-ok --json" in runs_blob
    assert "grep -q \"missing resource columns\"" in runs_blob
    assert "grep -q \"RHU resource profile\"" in runs_blob
    text = workflow.read_text(encoding="utf-8")
    assert "artifacts/runs/ci_legacy_resource/**" in text
    assert "artifacts/figures/ci_legacy_resource/**" in text


def test_report_preflight_workflow_has_legacy_ground_truth_metadata_path() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "legacy_ground_truth_metadata_path" in jobs

    legacy_job = jobs["legacy_ground_truth_metadata_path"]
    assert isinstance(legacy_job, dict)
    steps = legacy_job.get("steps", [])
    assert isinstance(steps, list)
    step_names = [str(step.get("name", "")) for step in steps if isinstance(step, dict)]

    assert "Drop ground-truth metadata fields to emulate legacy artifact" in step_names
    assert "Assert strict validation fails for legacy ground-truth artifact" in step_names
    assert "Capture legacy-compatible ground-truth validation warnings" in step_names
    assert "Assert report fails before regeneration" in step_names
    assert "Upload legacy ground-truth artifacts" in step_names

    runs_blob = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))
    assert "maxionbench validate --input artifacts/runs/ci_legacy_ground_truth --strict-schema --json" in runs_blob
    assert "maxionbench validate --input artifacts/runs/ci_legacy_ground_truth --legacy-ok --json" in runs_blob
    assert "grep -q \"missing ground truth metadata keys\"" in runs_blob
    assert "grep -q \"ground truth metadata\"" in runs_blob
    text = workflow.read_text(encoding="utf-8")
    assert "artifacts/runs/ci_legacy_ground_truth/**" in text
    assert "artifacts/figures/ci_legacy_ground_truth/**" in text


def test_report_preflight_workflow_enables_pip_cache() -> None:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)

    for job_name in (
        "conformance_readiness_gate",
        "report_preflight",
        "legacy_migration_path",
        "legacy_resource_profile_path",
        "legacy_ground_truth_metadata_path",
    ):
        job = jobs.get(job_name, {})
        assert isinstance(job, dict)
        steps = job.get("steps", [])
        assert isinstance(steps, list)
        setup_steps = [step for step in steps if isinstance(step, dict) and step.get("name") == "Setup Python"]
        assert len(setup_steps) == 1
        setup_with = setup_steps[0].get("with", {})
        assert isinstance(setup_with, dict)
        assert setup_with.get("cache") == "pip"
        cache_dep = str(setup_with.get("cache-dependency-path", ""))
        assert "pyproject.toml" in cache_dep


def test_branch_protection_docs_and_pr_template_reference_required_checks() -> None:
    policy = Path("docs/ci/branch_protection.md")
    template = Path(".github/pull_request_template.md")
    assert policy.exists()
    assert template.exists()

    policy_text = policy.read_text(encoding="utf-8")
    template_text = template.read_text(encoding="utf-8")

    check_primary = "report-preflight / report_preflight"
    check_conformance = "report-preflight / conformance_readiness_gate"
    check_legacy = "report-preflight / legacy_migration_path"
    check_legacy_resource = "report-preflight / legacy_resource_profile_path"
    check_legacy_ground_truth = "report-preflight / legacy_ground_truth_metadata_path"
    check_drift = "branch-protection-drift / verify_branch_protection"
    check_strict_readiness = "strict-readiness / strict_readiness_gate"
    check_publish_bundle = "publish-benchmark-bundle / publish_result_bundle"

    assert check_conformance in policy_text
    assert check_primary in policy_text
    assert check_legacy in policy_text
    assert check_legacy_resource in policy_text
    assert check_legacy_ground_truth in policy_text
    assert check_drift in policy_text
    assert check_strict_readiness in policy_text
    assert check_publish_bundle in policy_text
    assert check_conformance in template_text
    assert check_primary in template_text
    assert check_legacy in template_text
    assert check_legacy_resource in template_text
    assert check_legacy_ground_truth in template_text
    assert check_drift in template_text
    assert check_strict_readiness in template_text
    assert check_publish_bundle in template_text
    assert "maxionbench validate --input artifacts/runs --strict-schema --json" in template_text
