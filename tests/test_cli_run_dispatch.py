from __future__ import annotations

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.slurm import submit_plan as submit_plan_mod
from maxionbench.orchestration import runner as runner_mod
from maxionbench.tools import verify_branch_protection as verify_branch_mod
from maxionbench.tools import verify_slurm_plan as verify_slurm_plan_mod


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


def test_cli_verify_branch_protection_dispatches_optional_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 37

    monkeypatch.setattr(verify_branch_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-branch-protection",
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--timeout-s",
            "12.5",
            "--required-check",
            "report-preflight / report_preflight",
            "--include-drift-check",
            "--include-strict-readiness-check",
            "--include-publish-bundle-check",
            "--json",
        ]
    )
    assert code == 37
    assert captured["argv"] == [
        "--repo",
        "owner/repo",
        "--branch",
        "main",
        "--timeout-s",
        "12.5",
        "--required-check",
        "report-preflight / report_preflight",
        "--include-drift-check",
        "--include-strict-readiness-check",
        "--include-publish-bundle-check",
        "--json",
    ]


def test_cli_submit_slurm_plan_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 41

    monkeypatch.setattr(submit_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--seed",
            "123",
            "--skip-gpu",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 41
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--seed",
        "123",
        "--skip-gpu",
        "--dry-run",
        "--json",
    ]


def test_cli_verify_slurm_plan_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 43

    monkeypatch.setattr(verify_slurm_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--skip-gpu",
            "--json",
        ]
    )
    assert code == 43
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--skip-gpu",
        "--json",
    ]
