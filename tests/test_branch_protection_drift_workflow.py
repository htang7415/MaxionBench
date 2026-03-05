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

    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "maxionbench verify-branch-protection" in text
    assert '--repo "${GITHUB_REPOSITORY}"' in text
    assert "--branch main" in text
    assert "BRANCH_PROTECTION_TOKEN" in text
    assert "actions/upload-artifact@v4" in text
    assert "branch_protection_summary.json" in text
