from __future__ import annotations

from pathlib import Path


def test_readme_mentions_run_artifacts() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "`results.parquet`" in text
    assert "`run_metadata.json`" in text
    assert "`config_resolved.yaml`" in text
    assert "logs" in text
