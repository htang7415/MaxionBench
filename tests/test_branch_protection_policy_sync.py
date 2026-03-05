from __future__ import annotations

import re
from pathlib import Path

import yaml

from maxionbench.tools import verify_branch_protection as verify_mod


def _report_preflight_contexts() -> set[str]:
    workflow = Path(".github/workflows/report_preflight.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    return {f"report-preflight / {name}" for name in jobs.keys()}


def _branch_protection_required_contexts() -> set[str]:
    text = Path("docs/ci/branch_protection.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        start = lines.index("Required checks:")
    except ValueError:
        raise AssertionError("docs/ci/branch_protection.md missing `Required checks:` section")

    contexts: set[str] = set()
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Optional "):
            break
        if not stripped.startswith("- "):
            continue
        for match in re.findall(r"`([^`]+)`", stripped):
            contexts.add(match)
    return contexts


def _pr_template_checklist_contexts() -> set[str]:
    text = Path(".github/pull_request_template.md").read_text(encoding="utf-8")
    contexts: set[str] = set()
    for line in text.splitlines():
        if not line.strip().startswith("- [ ]"):
            continue
        for match in re.findall(r"`([^`]+)`", line):
            if " / " in match:
                contexts.add(match)
    return contexts


def _drift_workflow_required_check_contexts() -> set[str]:
    workflow = Path(".github/workflows/branch_protection_drift.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    job = jobs.get("verify_branch_protection", {})
    assert isinstance(job, dict)
    steps = job.get("steps", [])
    assert isinstance(steps, list)

    run_blob = "\n".join(
        str(step.get("run", ""))
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Verify branch protection required checks"
    )
    assert run_blob

    contexts = set(re.findall(r'--required-check\s+"([^"]+)"', run_blob))
    return contexts


def _drift_workflow_required_report_preflight_job_names() -> set[str]:
    names: set[str] = set()
    for context in _drift_workflow_required_check_contexts():
        prefix = "report-preflight / "
        if not context.startswith(prefix):
            continue
        names.add(context[len(prefix) :])
    return names


def test_branch_protection_doc_covers_report_preflight_jobs() -> None:
    workflow_contexts = _report_preflight_contexts()
    doc_contexts = _branch_protection_required_contexts()
    missing = sorted(workflow_contexts - doc_contexts)
    assert missing == []


def test_branch_protection_doc_mentions_policy_sync_guards() -> None:
    text = Path("docs/ci/branch_protection.md").read_text(encoding="utf-8")
    assert "## Automatic policy-sync guards" in text
    assert "tests/test_branch_protection_policy_sync.py" in text
    assert "tests/test_branch_protection_drift_workflow.py" in text
    assert "tests/test_report_preflight_workflow.py" in text
    assert "tests/test_report_figure_policy_sync.py" in text
    assert "maxionbench snapshot-required-checks" in text
    assert "artifacts/ci/required_checks_snapshot.json" in text


def test_pr_template_covers_report_preflight_jobs() -> None:
    workflow_contexts = _report_preflight_contexts()
    pr_contexts = _pr_template_checklist_contexts()
    missing = sorted(workflow_contexts - pr_contexts)
    assert missing == []


def test_verify_branch_protection_defaults_match_report_preflight_jobs() -> None:
    workflow_contexts = _report_preflight_contexts()
    assert set(verify_mod.DEFAULT_REQUIRED_CHECKS) == workflow_contexts


def test_drift_workflow_required_checks_match_defaults() -> None:
    contexts = _drift_workflow_required_check_contexts()
    assert contexts == set(verify_mod.DEFAULT_REQUIRED_CHECKS)


def test_drift_workflow_supports_optional_check_inputs() -> None:
    workflow = Path(".github/workflows/branch_protection_drift.yml")
    payload = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    on_block = payload.get("on", payload.get(True, {}))
    assert isinstance(on_block, dict)
    dispatch = on_block.get("workflow_dispatch", {})
    assert isinstance(dispatch, dict)
    inputs = dispatch.get("inputs", {})
    assert isinstance(inputs, dict)
    assert "include_strict_readiness_check" in inputs
    assert "include_publish_bundle_check" in inputs

    text = workflow.read_text(encoding="utf-8")
    assert "--include-strict-readiness-check" in text
    assert "--include-publish-bundle-check" in text


def test_drift_workflow_report_preflight_jobs_match_report_preflight_workflow() -> None:
    workflow_jobs = set()
    for context in _report_preflight_contexts():
        workflow_jobs.add(context.split(" / ", maxsplit=1)[1])
    assert _drift_workflow_required_report_preflight_job_names() == workflow_jobs
