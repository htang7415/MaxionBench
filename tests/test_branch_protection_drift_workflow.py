from __future__ import annotations

from pathlib import Path

import yaml


def test_branch_protection_drift_workflow_has_expected_schedule_and_command() -> None:
    workflow = Path(".github/workflows/branch_protection_drift.yml")
    assert workflow.exists()

    text = workflow.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert "jobs" in payload
    assert "verify_branch_protection" in payload["jobs"]
    on_block = payload.get("on", payload.get(True, {}))
    assert isinstance(on_block, dict)
    assert "workflow_dispatch" in on_block
    dispatch = on_block["workflow_dispatch"]
    assert isinstance(dispatch, dict)
    inputs = dispatch.get("inputs", {})
    assert isinstance(inputs, dict)
    assert "include_strict_readiness_check" in inputs
    assert "include_publish_bundle_check" in inputs

    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "maxionbench verify-branch-protection" in text
    assert '--repo "${GITHUB_REPOSITORY}"' in text
    assert "--branch main" in text
    assert '--required-check "report-preflight / conformance_readiness_gate"' in text
    assert '--required-check "report-preflight / report_preflight"' in text
    assert '--required-check "report-preflight / legacy_migration_path"' in text
    assert '--required-check "report-preflight / legacy_resource_profile_path"' in text
    assert '--required-check "report-preflight / legacy_ground_truth_metadata_path"' in text
    assert '--include-strict-readiness-check' in text
    assert '--include-publish-bundle-check' in text
    assert 'if [[ "${{ inputs.include_strict_readiness_check }}" == "true" ]]; then' in text
    assert 'if [[ "${{ inputs.include_publish_bundle_check }}" == "true" ]]; then' in text
    assert "BRANCH_PROTECTION_TOKEN" in text
    assert "actions/upload-artifact@v4" in text
    assert "branch_protection_summary.json" in text
