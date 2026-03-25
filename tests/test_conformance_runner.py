from __future__ import annotations

import json
import os
import subprocess

import pytest

from maxionbench.conformance import run as conformance_run_mod


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
