from __future__ import annotations

from pathlib import Path


def test_readme_documents_report_output_policy_inspection_workflow() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Report sidecar output-policy inspection:" in text
    assert "maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 --strict --json" in text
    assert "--output artifacts/ci/report_output_policy_summary.json" in text
    assert '"pass": true' in text
    assert "output_path_class_counts" in text
    assert "rerun inspection with `--strict --json` and verify `error_count: 0`" in text
    assert "Strict mode exit codes for policy audits:" in text
    assert "snapshot-required-checks --strict ...` exits with code `2`" in text
    assert "inspect-report-output-policy --strict ...` exits with code `2`" in text


def test_readme_premerge_automation_mentions_output_policy_enforcement() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "verify-behavior-cards --behavior-dir docs/behavior --json" in text
    assert "enforces report output-policy sidecar checks" in text
    assert "maxionbench inspect-report-output-policy --strict" in text
    assert "artifacts/ci/report_output_policy_summary.json" in text


def test_readme_has_output_policy_troubleshooting_table() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Common `inspect-report-output-policy` failures and fixes:" in text
    assert "| Summary key | Typical meaning | Fix action |" in text
    assert "`missing_output_policy_files`" in text
    assert "`invalid_output_policy`" in text
    assert "`invalid_json_files`" in text
    assert "`no_meta_files`" in text
    assert "`mixed_output_path_classes` / `mixed_modes` / `mixed_milestone_ids`" in text
    assert "`output_path_class_mismatch` / `milestone_id_mismatch`" in text
    assert "`resolved_out_dir_mismatch` / `milestone_root_mismatch`" in text
