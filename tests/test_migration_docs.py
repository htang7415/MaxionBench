from __future__ import annotations

from pathlib import Path


def test_readme_indexes_source_of_truth_docs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "1. `project.md`" in text
    assert "2. `prompt.md`" in text
    assert "3. `document.md`" in text
    assert "4. `command.md`" in text
    assert "5. `command-mac.md`" in text
