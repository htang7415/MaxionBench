from __future__ import annotations

from pathlib import Path


def test_readme_mentions_optional_protocol_enforcement_validation() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "optional pinned protocol audit for paper-grade runs" in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json" in text
