from __future__ import annotations

import subprocess
from pathlib import Path


def test_gitignore_blocks_python_cache_artifacts() -> None:
    payload = Path(".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in payload
    assert "*.py[cod]" in payload
    assert ".pytest_cache/" in payload
    assert "*.md" in payload
    assert "*.sh" in payload
    assert "!README.md" in payload
    assert "build/" in payload
    assert "artifacts/containers/" in payload
    assert "artifacts/workstation_runs/" in payload
    assert "results/" in payload
    assert "results_quick_check/" in payload
    assert ".env.slurm.*" not in payload
    assert "prepare_containers.sh" not in payload


def test_repository_has_no_tracked_python_cache_artifacts() -> None:
    proc = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    banned = [path for path in tracked if "__pycache__/" in path or path.endswith(".pyc")]
    assert banned == []


def test_local_docs_and_scripts_are_gitignored_except_readme() -> None:
    ignored_paths = [
        "AGENTS.md",
        "CLAUDE.md",
        "command-mac.md",
        "command.md",
        "document.md",
        "preprocess_all_datasets.sh",
        "project.md",
        "prompt.md",
    ]
    for path in ignored_paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{path} should be ignored by git"

    proc = subprocess.run(
        ["git", "check-ignore", "-q", "README.md"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, "README.md should not be ignored by git"
