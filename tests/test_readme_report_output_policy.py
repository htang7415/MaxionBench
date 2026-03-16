from __future__ import annotations

from pathlib import Path


def test_readme_documents_report_outputs_and_commands() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "artifacts/figures/milestones/Mx/" in text
    assert "artifacts/figures/final/" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1" in text
    assert "python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final" in text
