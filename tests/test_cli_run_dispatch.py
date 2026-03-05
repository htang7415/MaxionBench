from __future__ import annotations

from maxionbench.cli import main as cli_main
from maxionbench.orchestration import runner as runner_mod


def test_cli_run_dispatches_readiness_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 31

    monkeypatch.setattr(runner_mod, "main", _fake_main)
    code = cli_main(
        [
            "run",
            "--config",
            "configs/scenarios/s1_ann_frontier.yaml",
            "--seed",
            "42",
            "--repeats",
            "3",
            "--no-retry",
            "--output-dir",
            "artifacts/runs/dispatch",
            "--d3-params",
            "artifacts/calibration/d3_params.yaml",
            "--enforce-readiness",
            "--conformance-matrix",
            "artifacts/conformance/conformance_matrix.csv",
            "--behavior-dir",
            "docs/behavior",
            "--allow-gpu-unavailable",
        ]
    )
    assert code == 31
    assert captured["argv"] == [
        "--config",
        "configs/scenarios/s1_ann_frontier.yaml",
        "--seed",
        "42",
        "--repeats",
        "3",
        "--no-retry",
        "--output-dir",
        "artifacts/runs/dispatch",
        "--d3-params",
        "artifacts/calibration/d3_params.yaml",
        "--enforce-readiness",
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
        "--allow-gpu-unavailable",
    ]
