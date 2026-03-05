from __future__ import annotations

import re
from pathlib import Path


def _assert_common_commands(text: str) -> None:
    assert "maxionbench verify-pins --config-dir configs/scenarios --json" in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --json" in text
    assert "maxionbench validate --input artifacts/runs --legacy-ok --json" in text
    assert "maxionbench migrate-stage-timing --input artifacts/runs --dry-run" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones/Mx" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --milestone-id M3" in text
    assert "maxionbench snapshot-required-checks --output artifacts/ci/required_checks_snapshot.json --strict --json" in text


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
