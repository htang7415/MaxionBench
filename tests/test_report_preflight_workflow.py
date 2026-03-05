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
    verify_hygiene_cmd = "pytest -q tests/test_repo_hygiene.py"
    verify_command_docs_cmd = "pytest -q tests/test_command_docs.py"
    verify_migration_docs_cmd = "pytest -q tests/test_migration_docs.py"
    report_cmd = "maxionbench report"
    assert verify_pins_cmd in text
    assert verify_hygiene_cmd in text
    assert verify_command_docs_cmd in text
    assert verify_migration_docs_cmd in text
    assert validate_cmd in text
    assert report_cmd in text
    assert text.index(verify_pins_cmd) < text.index(validate_cmd)
    assert text.index(verify_hygiene_cmd) < text.index(validate_cmd)
    assert text.index(verify_command_docs_cmd) < text.index(validate_cmd)
    assert text.index(verify_migration_docs_cmd) < text.index(validate_cmd)
    assert text.index(validate_cmd) < text.index(report_cmd)

    upload_marker = "uses: actions/upload-artifact@v4"
    assert upload_marker in text
    assert "if: always()" in text
    assert "artifacts/runs/ci_preflight/**" in text
    assert "artifacts/figures/ci_preflight/**" in text


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
    check_legacy = "report-preflight / legacy_migration_path"
    check_legacy_resource = "report-preflight / legacy_resource_profile_path"
    check_legacy_ground_truth = "report-preflight / legacy_ground_truth_metadata_path"
    check_drift = "branch-protection-drift / verify_branch_protection"

    assert check_primary in policy_text
    assert check_legacy in policy_text
    assert check_legacy_resource in policy_text
    assert check_legacy_ground_truth in policy_text
    assert check_drift in policy_text
    assert check_primary in template_text
    assert check_legacy in template_text
    assert check_legacy_resource in template_text
    assert check_legacy_ground_truth in template_text
    assert check_drift in template_text
    assert "maxionbench validate --input artifacts/runs --strict-schema --json" in template_text
