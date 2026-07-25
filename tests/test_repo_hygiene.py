from __future__ import annotations

import subprocess
from pathlib import Path


PUBLIC_TEXT_EXTENSIONS = {
    ".bib",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonld",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_NEEDLES = (
    "/" + "Users/",
    "/" + "Volumes/" + "Max",
    "/" + "home/data/",
)


def test_gitignore_blocks_generated_and_local_artifacts() -> None:
    payload = Path(".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in payload
    assert "*.py[cod]" in payload
    assert ".pytest_cache/" in payload
    assert "*.md" not in payload
    assert "*.sh" not in payload
    assert "build/" in payload
    assert "artifacts/" in payload
    assert "results/" in payload
    assert "results_quick_check/" in payload
    assert ".codex" in payload
    assert ".vscode/" in payload
    assert ".agents/" in payload
    assert ".claude/" in payload
    assert "AGENTS.md" not in payload
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


def test_only_explicit_local_docs_are_gitignored() -> None:
    ignored_paths = [
        "CLAUDE.md",
        "command.md",
    ]
    for path in ignored_paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{path} should be ignored by git"

    visible_paths = [
        "AGENTS.md",
        "README.md",
        "docs/behavior/_template.md",
    ]
    for path in visible_paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, f"{path} should not be ignored by git"


def test_tracked_public_text_has_no_local_identity_leaks() -> None:
    proc = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    offenders: list[str] = []
    for path in tracked:
        if not path.exists():
            continue
        if path.suffix not in PUBLIC_TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in PRIVATE_NEEDLES:
            if needle in text:
                offenders.append(f"{path}:{needle}")
    assert offenders == []
