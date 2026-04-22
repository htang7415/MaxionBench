from __future__ import annotations

from pathlib import Path


def test_readme_points_mac_users_to_reduced_lane_doc() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "command.md" not in text
    assert "## Source of truth" not in text


def test_dockerfile_omits_local_only_command_docs() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")

    assert "command.md" not in text
    assert "command-mac.md" not in text
