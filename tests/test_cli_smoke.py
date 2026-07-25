from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_cli_help_only_needs_standard_library() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "maxionbench.cli", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "usage: maxionbench" in completed.stdout
