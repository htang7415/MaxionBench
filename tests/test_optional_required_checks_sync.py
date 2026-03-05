from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.tools import verify_branch_protection as verify_mod


def _workflow_context(path: Path, *, job_name: str) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    workflow_name = payload.get("name")
    assert isinstance(workflow_name, str) and workflow_name
    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert job_name in jobs
    return f"{workflow_name} / {job_name}"


def test_optional_required_check_constants_match_workflow_contexts() -> None:
    strict_context = _workflow_context(
        Path(".github/workflows/strict_readiness.yml"),
        job_name="strict_readiness_gate",
    )
    publish_context = _workflow_context(
        Path(".github/workflows/publish_benchmark_bundle.yml"),
        job_name="publish_result_bundle",
    )
    assert verify_mod.OPTIONAL_STRICT_READINESS_CHECK == strict_context
    assert verify_mod.OPTIONAL_PUBLISH_BUNDLE_CHECK == publish_context
    assert verify_mod.OPTIONAL_REQUIRED_CHECKS == (
        verify_mod.OPTIONAL_DRIFT_CHECK,
        verify_mod.OPTIONAL_STRICT_READINESS_CHECK,
        verify_mod.OPTIONAL_PUBLISH_BUNDLE_CHECK,
    )


def test_optional_required_checks_are_documented_in_policy_files() -> None:
    policy_text = Path("docs/ci/branch_protection.md").read_text(encoding="utf-8")
    template_text = Path(".github/pull_request_template.md").read_text(encoding="utf-8")
    for context in verify_mod.OPTIONAL_REQUIRED_CHECKS:
        assert context in policy_text
        assert context in template_text
