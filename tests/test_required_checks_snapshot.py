from __future__ import annotations

import json
from pathlib import Path

from maxionbench.cli import main as cli_main
from maxionbench.tools import required_checks_snapshot as snapshot_mod
from maxionbench.tools.required_checks_snapshot import build_required_checks_snapshot


def test_build_required_checks_snapshot_passes_on_repo_defaults() -> None:
    snapshot = build_required_checks_snapshot(
        report_workflow_path=Path(".github/workflows/report_preflight.yml"),
        drift_workflow_path=Path(".github/workflows/branch_protection_drift.yml"),
        branch_protection_doc_path=Path("docs/ci/branch_protection.md"),
        pr_template_path=Path(".github/pull_request_template.md"),
    )
    assert snapshot["pass"] is True
    checks = snapshot["checks"]
    assert checks["jobs_vs_defaults"] is True
    assert checks["jobs_vs_drift_workflow"] is True
    assert checks["jobs_vs_branch_protection_doc"] is True
    assert checks["jobs_vs_pr_template"] is True
    assert checks["pr_template_optional_contexts_valid"] is True


def test_snapshot_required_checks_cli_writes_artifact(tmp_path: Path) -> None:
    out_path = tmp_path / "required_checks_snapshot.json"
    code = cli_main(["snapshot-required-checks", "--output", str(out_path), "--json"])
    assert code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["required_check_contexts"]["from_report_preflight_jobs"]


def test_snapshot_required_checks_strict_mode_fails_on_mismatch(tmp_path: Path) -> None:
    doc_src = Path("docs/ci/branch_protection.md")
    text = doc_src.read_text(encoding="utf-8")
    mutated = text.replace("- `report-preflight / legacy_resource_profile_path`\n", "")
    doc_mut = tmp_path / "branch_protection.md"
    doc_mut.write_text(mutated, encoding="utf-8")
    out_path = tmp_path / "required_checks_snapshot.json"

    code = cli_main(
        [
            "snapshot-required-checks",
            "--output",
            str(out_path),
            "--branch-protection-doc",
            str(doc_mut),
            "--strict",
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["pass"] is False
    assert payload["checks"]["jobs_vs_branch_protection_doc"] is False


def test_snapshot_required_checks_cli_dispatches_arguments(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 29

    monkeypatch.setattr(snapshot_mod, "main", _fake_main)
    code = cli_main(
        [
            "snapshot-required-checks",
            "--output",
            "artifacts/ci/snapshot.json",
            "--report-workflow",
            ".github/workflows/report_preflight.yml",
            "--drift-workflow",
            ".github/workflows/branch_protection_drift.yml",
            "--branch-protection-doc",
            "docs/ci/branch_protection.md",
            "--pr-template",
            ".github/pull_request_template.md",
            "--strict",
            "--json",
        ]
    )
    assert code == 29
    assert captured["argv"] == [
        "--output",
        "artifacts/ci/snapshot.json",
        "--report-workflow",
        ".github/workflows/report_preflight.yml",
        "--drift-workflow",
        ".github/workflows/branch_protection_drift.yml",
        "--branch-protection-doc",
        "docs/ci/branch_protection.md",
        "--pr-template",
        ".github/pull_request_template.md",
        "--strict",
        "--json",
    ]


def test_snapshot_required_checks_strict_mode_fails_on_unexpected_optional_pr_context(tmp_path: Path) -> None:
    template_src = Path(".github/pull_request_template.md")
    text = template_src.read_text(encoding="utf-8")
    mutated = text + "\n- [ ] `random-workflow / unknown_gate` passed (if enforced)\n"
    template_mut = tmp_path / "pull_request_template.md"
    template_mut.write_text(mutated, encoding="utf-8")
    out_path = tmp_path / "required_checks_snapshot.json"

    code = cli_main(
        [
            "snapshot-required-checks",
            "--output",
            str(out_path),
            "--pr-template",
            str(template_mut),
            "--strict",
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["pass"] is False
    assert payload["checks"]["jobs_vs_pr_template"] is False
    assert payload["checks"]["pr_template_optional_contexts_valid"] is False
    assert payload["diff"]["unexpected_optional_vs_pr_template"] == ["random-workflow / unknown_gate"]
