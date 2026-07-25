from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from maxionbench.conformance import run as conformance_run_mod


def test_conformance_runner_mock_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
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


def test_conformance_runner_expands_env_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAXIONBENCH_QDRANT_HOST", "qdrant")
    captured: dict[str, object] = {}

    def _fake_pytest_main(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        captured["options"] = json.loads(os.environ["MAXIONBENCH_CONFORMANCE_ADAPTER_OPTIONS_JSON"])
        return 0

    monkeypatch.setattr(conformance_run_mod.pytest, "main", _fake_pytest_main)
    code = conformance_run_mod.main(
        [
            "--adapter",
            "qdrant",
            "--adapter-options-json",
            '{"host":"${MAXIONBENCH_QDRANT_HOST:-127.0.0.1}","port":"${MAXIONBENCH_QDRANT_PORT:-6333}"}',
        ]
    )
    assert code == 0
    assert captured["options"] == {"host": "qdrant", "port": "6333"}
    assert "-s" in captured["argv"]


def test_conformance_runner_resolves_suite_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def _fake_pytest_main(argv: list[str]) -> int:
        captured.extend(argv)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(conformance_run_mod.pytest, "main", _fake_pytest_main)

    assert conformance_run_mod.main(["--adapter", "mock"]) == 0
    suite_path = Path(captured[-1])
    assert suite_path.is_absolute()
    assert suite_path.name == "test_conformance.py"
    assert suite_path.exists()
