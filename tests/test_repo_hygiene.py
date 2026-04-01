from __future__ import annotations

import subprocess
from pathlib import Path


def test_gitignore_blocks_python_cache_artifacts() -> None:
    payload = Path(".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in payload
    assert "*.py[cod]" in payload
    assert ".pytest_cache/" in payload
    assert "artifacts/containers/" in payload
    assert "artifacts/workstation_runs/" in payload
    assert "results/" in payload
    assert "!run_workstation.sh" in payload
    assert "!run_docker_scenario.sh" in payload
    assert "!save_results_bundle.sh" in payload
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
