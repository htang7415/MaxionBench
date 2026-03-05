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
            "report-preflight / conformance_readiness_gate",
            "report-preflight / report_preflight",
            "report-preflight / legacy_migration_path",
            "report-preflight / legacy_resource_profile_path",
            "report-preflight / legacy_ground_truth_metadata_path",
        ],
    )
    assert summary["pass"] is False
    assert set(summary["missing_checks"]) == {
        "report-preflight / conformance_readiness_gate",
        "report-preflight / legacy_migration_path",
        "report-preflight / legacy_resource_profile_path",
        "report-preflight / legacy_ground_truth_metadata_path",
    }


def test_resolve_required_checks_optionally_includes_extra_checks() -> None:
    checks = verify_mod.resolve_required_checks(
        None,
        include_drift_check=False,
        include_strict_readiness_check=False,
        include_publish_bundle_check=False,
    )
    assert "report-preflight / conformance_readiness_gate" in checks
    assert "report-preflight / report_preflight" in checks
    assert "report-preflight / legacy_migration_path" in checks
    assert "report-preflight / legacy_resource_profile_path" in checks
    assert "report-preflight / legacy_ground_truth_metadata_path" in checks
    assert verify_mod.OPTIONAL_DRIFT_CHECK not in checks
    assert verify_mod.OPTIONAL_STRICT_READINESS_CHECK not in checks
    assert verify_mod.OPTIONAL_PUBLISH_BUNDLE_CHECK not in checks

    checks_with_optional = verify_mod.resolve_required_checks(
        None,
        include_drift_check=True,
        include_strict_readiness_check=True,
        include_publish_bundle_check=True,
    )
    assert verify_mod.OPTIONAL_DRIFT_CHECK in checks_with_optional
    assert verify_mod.OPTIONAL_STRICT_READINESS_CHECK in checks_with_optional
    assert verify_mod.OPTIONAL_PUBLISH_BUNDLE_CHECK in checks_with_optional


def test_verify_main_and_cli_return_expected_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    passing_payload = {
        "required_status_checks": {
            "contexts": [
                "report-preflight / conformance_readiness_gate",
                "report-preflight / report_preflight",
                "report-preflight / legacy_migration_path",
                "report-preflight / legacy_resource_profile_path",
                "report-preflight / legacy_ground_truth_metadata_path",
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
                "report-preflight / conformance_readiness_gate",
                "report-preflight / report_preflight",
                "report-preflight / legacy_migration_path",
                "report-preflight / legacy_resource_profile_path",
                "report-preflight / legacy_ground_truth_metadata_path",
                verify_mod.OPTIONAL_DRIFT_CHECK,
                verify_mod.OPTIONAL_STRICT_READINESS_CHECK,
                verify_mod.OPTIONAL_PUBLISH_BUNDLE_CHECK,
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

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: passing_payload,
    )
    strict_missing = cli_main(
        ["verify-branch-protection", "--repo", "owner/repo", "--include-strict-readiness-check", "--json"]
    )
    assert strict_missing == 2

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: passing_payload,
    )
    publish_missing = cli_main(
        ["verify-branch-protection", "--repo", "owner/repo", "--include-publish-bundle-check", "--json"]
    )
    assert publish_missing == 2

    monkeypatch.setattr(
        verify_mod,
        "fetch_branch_protection",
        lambda repo, branch, token, timeout_s: drift_passing_payload,
    )
    strict_publish_ok = cli_main(
        [
            "verify-branch-protection",
            "--repo",
            "owner/repo",
            "--include-strict-readiness-check",
            "--include-publish-bundle-check",
            "--json",
        ]
    )
    assert strict_publish_ok == 0
