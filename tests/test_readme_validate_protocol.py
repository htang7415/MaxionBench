from __future__ import annotations

from pathlib import Path


def test_readme_mentions_validation_command() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "python -m maxionbench.cli validate --input artifacts/runs --strict-schema --json" in text
