from __future__ import annotations

import subprocess


def test_conformance_runner_mock_cli() -> None:
    completed = subprocess.run(
        [
            "python",
            "-m",
            "maxionbench.conformance.run",
            "--adapter",
            "mock",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
