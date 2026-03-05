from __future__ import annotations

import json

import pytest

from maxionbench.cli import main as cli_main
from maxionbench.tools import verify_branch_protection as verify_mod


def test_extract_required_check_contexts_supports_contexts_and_checks() -> None:
    payload = {
        "required_status_checks": {
            "contexts": ["a", "b"],
            "checks": [{"context": "c"}, {"context": "b"}],
        }
    }
    contexts = verify_mod.extract_required_check_contexts(payload)
    assert contexts == {"a", "b", "c"}


def test_evaluate_branch_protection_detects_missing_checks() -> None:
    payload = {"required_status_checks": {"contexts": ["report-preflight / report_preflight"]}}
    summary = verify_mod.evaluate_branch_protection(
        payload,
        required_checks=[
            "report-preflight / report_preflight",
            "report-preflight / legacy_migration_path",
        ],
    )
    assert summary["pass"] is False
    assert summary["missing_checks"] == ["report-preflight / legacy_migration_path"]


def test_resolve_required_checks_optionally_includes_drift_check() -> None:
    checks = verify_mod.resolve_required_checks(None, include_drift_check=False)
    assert "report-preflight / report_preflight" in checks
    assert "report-preflight / legacy_migration_path" in checks
    assert verify_mod.OPTIONAL_DRIFT_CHECK not in checks

    checks_with_drift = verify_mod.resolve_required_checks(None, include_drift_check=True)
    assert verify_mod.OPTIONAL_DRIFT_CHECK in checks_with_drift


def test_verify_main_and_cli_return_expected_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    passing_payload = {
        "required_status_checks": {
            "contexts": [
                "report-preflight / report_preflight",
                "report-preflight / legacy_migration_path",
            ]
        }
    }
    failing_payload = {"required_status_checks": {"contexts": ["report-preflight / report_preflight"]}}

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: passing_payload,
    )
    code = verify_mod.main(["--repo", "owner/repo", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["pass"] is True

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: failing_payload,
    )
    code = cli_main(["verify-branch-protection", "--repo", "owner/repo", "--json"])
    assert code == 2

    drift_passing_payload = {
        "required_status_checks": {
            "contexts": [
                "report-preflight / report_preflight",
                "report-preflight / legacy_migration_path",
                verify_mod.OPTIONAL_DRIFT_CHECK,
            ]
        }
    }
    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: passing_payload,
    )
    drift_missing = cli_main(
        ["verify-branch-protection", "--repo", "owner/repo", "--include-drift-check", "--json"]
    )
    assert drift_missing == 2

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: drift_passing_payload,
    )
    drift_ok = cli_main(["verify-branch-protection", "--repo", "owner/repo", "--include-drift-check", "--json"])
    assert drift_ok == 0
