from __future__ import annotations

import re
from pathlib import Path


def _assert_common_commands(text: str) -> None:
    assert "maxionbench verify-pins --config-dir configs/scenarios --json" in text
    assert "maxionbench verify-behavior-cards --behavior-dir docs/behavior --json" in text
    assert "--enforce-readiness" in text
    assert "--conformance-matrix artifacts/conformance/conformance_matrix.csv" in text
    assert "--behavior-dir docs/behavior" in text
    assert "--allow-gpu-unavailable" in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --json"
    ) in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --allow-gpu-unavailable --json"
    ) in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --allow-gpu-unavailable --allow-nonpass-status --json"
    ) in text
    assert (
        "maxionbench pre-run-gate --config configs/scenarios/s1_ann_frontier_qdrant.yaml "
        "--conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior --json"
    ) in text
    assert (
        "maxionbench pre-run-gate --config configs/scenarios/s1_ann_frontier_qdrant.yaml "
        "--conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior "
        "--allow-gpu-unavailable --json"
    ) in text
    assert (
        "maxionbench verify-promotion-gate --strict-readiness-summary "
        "artifacts/conformance_strict/engine_readiness_summary.json --json"
    ) in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --json" in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json" in text
    assert "maxionbench validate --input artifacts/runs --legacy-ok --json" in text
    assert "maxionbench migrate-stage-timing --input artifacts/runs --dry-run" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones/Mx" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --milestone-id M3" in text
    assert "maxionbench snapshot-required-checks --output artifacts/ci/required_checks_snapshot.json --strict --json" in text
    assert "maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 --strict --json" in text
    assert (
        "maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 "
        "--output artifacts/ci/report_output_policy_summary.json --strict --json"
    ) in text
    assert "exits with code `2` when mismatches are detected" in text
    assert "exits with code `2` when sidecar policy checks fail" in text
    assert "Preflight CI writes both policy artifacts together:" in text
    assert "`artifacts/ci/required_checks_snapshot.json`" in text
    assert "`artifacts/ci/report_output_policy_summary.json`" in text


def _assert_no_stale_validate_invocation(text: str) -> None:
    stale_pattern = re.compile(r"maxionbench validate --input artifacts/runs --json")
    assert stale_pattern.search(text) is None


def test_command_docs_use_strict_validate_and_legacy_mode() -> None:
    command_md = Path("command.md")
    command_mac_md = Path("command-mac.md")
    assert command_md.exists()
    assert command_mac_md.exists()

    text_command = command_md.read_text(encoding="utf-8")
    text_mac = command_mac_md.read_text(encoding="utf-8")

    _assert_common_commands(text_command)
    _assert_common_commands(text_mac)
    _assert_no_stale_validate_invocation(text_command)
    _assert_no_stale_validate_invocation(text_mac)
