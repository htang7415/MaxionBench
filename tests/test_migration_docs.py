from __future__ import annotations

from pathlib import Path


def test_readme_links_command_docs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[command.md]" in text
    assert "[command-mac.md]" in text
    assert "## Source of truth" not in text
