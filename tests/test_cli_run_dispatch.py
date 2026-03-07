from __future__ import annotations

from maxionbench.cli import main as cli_main
from maxionbench.conformance import matrix as conformance_matrix_mod
from maxionbench.orchestration.slurm import submit_plan as submit_plan_mod
from maxionbench.orchestration import runner as runner_mod
from maxionbench.tools import ci_protocol_audit as ci_protocol_audit_mod
from maxionbench.tools import verify_branch_protection as verify_branch_mod
from maxionbench.tools import verify_conformance_configs as verify_conformance_configs_mod
from maxionbench.tools import verify_d3_calibration as verify_d3_calibration_mod
from maxionbench.tools import verify_dataset_manifests as verify_dataset_manifests_mod
from maxionbench.tools import verify_engine_readiness as verify_engine_readiness_mod
from maxionbench.tools import verify_pins as verify_pins_mod
from maxionbench.tools import verify_promotion_gate as verify_promotion_gate_mod
from maxionbench.tools import verify_slurm_plan as verify_slurm_plan_mod
from maxionbench.tools import validate_slurm_snapshots as validate_slurm_snapshots_mod


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


def test_cli_conformance_matrix_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 39

    monkeypatch.setattr(conformance_matrix_mod, "main", _fake_main)
    code = cli_main(
        [
            "conformance-matrix",
            "--config-dir",
            "configs/conformance",
            "--out-dir",
            "artifacts/conformance",
            "--timeout-s",
            "45.5",
        ]
    )
    assert code == 39
    assert captured["argv"] == [
        "--config-dir",
        "configs/conformance",
        "--out-dir",
        "artifacts/conformance",
        "--timeout-s",
        "45.5",
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


def test_cli_submit_slurm_plan_dispatches_scenario_config_dir(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 141

    monkeypatch.setattr(submit_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--seed",
            "42",
            "--scenario-config-dir",
            "configs/scenarios_paper",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 141
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--seed",
        "42",
        "--scenario-config-dir",
        "configs/scenarios_paper",
        "--dry-run",
        "--json",
    ]


def test_cli_submit_slurm_plan_dispatches_slurm_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 241

    monkeypatch.setattr(submit_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--seed",
            "42",
            "--slurm-profile",
            "your_cluster",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 241
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--seed",
        "42",
        "--slurm-profile",
        "your_cluster",
        "--dry-run",
        "--json",
    ]


def test_cli_submit_slurm_plan_dispatches_output_root(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 242

    monkeypatch.setattr(submit_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--seed",
            "42",
            "--output-root",
            "artifacts/workstation_runs/example/results/slurm",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 242
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--seed",
        "42",
        "--output-root",
        "artifacts/workstation_runs/example/results/slurm",
        "--dry-run",
        "--json",
    ]


def test_cli_submit_slurm_plan_dispatches_container_runtime_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 243

    monkeypatch.setattr(submit_plan_mod, "main", _fake_main)
    code = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--seed",
            "42",
            "--container-runtime",
            "apptainer",
            "--container-image",
            "/shared/containers/maxionbench.sif",
            "--container-bind",
            "/shared/datasets",
            "--container-bind",
            "/shared/models/hf:/shared/models/hf",
            "--hf-cache-dir",
            "/shared/models/hf",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 243
    assert captured["argv"] == [
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--seed",
        "42",
        "--container-runtime",
        "apptainer",
        "--container-image",
        "/shared/containers/maxionbench.sif",
        "--container-bind",
        "/shared/datasets",
        "--container-bind",
        "/shared/models/hf:/shared/models/hf",
        "--hf-cache-dir",
        "/shared/models/hf",
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


def test_cli_verify_dataset_manifests_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 45

    monkeypatch.setattr(verify_dataset_manifests_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-dataset-manifests",
            "--manifest-dir",
            "maxionbench/datasets/manifests",
            "--json",
        ]
    )
    assert code == 45
    assert captured["argv"] == [
        "--manifest-dir",
        "maxionbench/datasets/manifests",
        "--json",
    ]


def test_cli_verify_conformance_configs_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 52

    monkeypatch.setattr(verify_conformance_configs_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-conformance-configs",
            "--config-dir",
            "configs/conformance",
            "--allow-gpu-unavailable",
            "--json",
        ]
    )
    assert code == 52
    assert captured["argv"] == [
        "--config-dir",
        "configs/conformance",
        "--allow-gpu-unavailable",
        "--json",
    ]


def test_cli_verify_pins_dispatches_dev_scale_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 48

    monkeypatch.setattr(verify_pins_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-pins",
            "--config-dir",
            "configs/scenarios",
            "--allow-dev-calibrate-d3-scale",
            "--json",
        ]
    )
    assert code == 48
    assert captured["argv"] == [
        "--config-dir",
        "configs/scenarios",
        "--allow-dev-calibrate-d3-scale",
        "--json",
    ]


def test_cli_verify_pins_dispatches_strict_d3_scale_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 148

    monkeypatch.setattr(verify_pins_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-pins",
            "--config-dir",
            "configs/scenarios",
            "--strict-d3-scenario-scale",
            "--json",
        ]
    )
    assert code == 148
    assert captured["argv"] == [
        "--config-dir",
        "configs/scenarios",
        "--strict-d3-scenario-scale",
        "--json",
    ]


def test_cli_verify_engine_readiness_dispatches_require_mock_pass(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 49

    monkeypatch.setattr(verify_engine_readiness_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            "artifacts/conformance/conformance_matrix.csv",
            "--behavior-dir",
            "docs/behavior",
            "--allow-gpu-unavailable",
            "--allow-nonpass-status",
            "--require-mock-pass",
            "--json",
        ]
    )
    assert code == 49
    assert captured["argv"] == [
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
        "--allow-gpu-unavailable",
        "--allow-nonpass-status",
        "--require-mock-pass",
        "--json",
    ]


def test_cli_verify_promotion_gate_dispatches_conformance_matrix(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 59

    monkeypatch.setattr(verify_promotion_gate_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            "artifacts/promotion/engine_readiness_summary.json",
            "--conformance-matrix",
            "artifacts/promotion/conformance_matrix.csv",
            "--json",
        ]
    )
    assert code == 59
    assert captured["argv"] == [
        "--strict-readiness-summary",
        "artifacts/promotion/engine_readiness_summary.json",
        "--conformance-matrix",
        "artifacts/promotion/conformance_matrix.csv",
        "--json",
    ]


def test_cli_verify_d3_calibration_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 46

    monkeypatch.setattr(verify_d3_calibration_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-d3-calibration",
            "--d3-params",
            "artifacts/calibration/d3_params.yaml",
            "--min-vectors",
            "10000000",
            "--strict",
            "--json",
        ]
    )
    assert code == 46
    assert captured["argv"] == [
        "--d3-params",
        "artifacts/calibration/d3_params.yaml",
        "--min-vectors",
        "10000000",
        "--strict",
        "--json",
    ]


def test_cli_validate_slurm_snapshots_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 47

    monkeypatch.setattr(validate_slurm_snapshots_mod, "main", _fake_main)
    code = cli_main(
        [
            "validate-slurm-snapshots",
            "--verify-path",
            "artifacts/ci/slurm_plan_verify.json",
            "--verify-path",
            "artifacts/ci/slurm_plan_verify_skip_gpu.json",
            "--submit-path",
            "artifacts/ci/slurm_submit_plan_dry_run.json",
            "--submit-path",
            "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json",
            "--required-baseline-scenario",
            "configs/scenarios/s1_ann_frontier_d3.yaml",
            "--json",
        ]
    )
    assert code == 47
    assert captured["argv"] == [
        "--required-baseline-scenario",
        "configs/scenarios/s1_ann_frontier_d3.yaml",
        "--verify-path",
        "artifacts/ci/slurm_plan_verify.json",
        "--verify-path",
        "artifacts/ci/slurm_plan_verify_skip_gpu.json",
        "--submit-path",
        "artifacts/ci/slurm_submit_plan_dry_run.json",
        "--submit-path",
        "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json",
        "--json",
    ]


def test_cli_ci_protocol_audit_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 53

    monkeypatch.setattr(ci_protocol_audit_mod, "main", _fake_main)
    code = cli_main(
        [
            "ci-protocol-audit",
            "--config-dir",
            "configs/scenarios",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--manifest-dir",
            "maxionbench/datasets/manifests",
            "--verify-path",
            "artifacts/ci/slurm_plan_verify.json",
            "--verify-path",
            "artifacts/ci/slurm_plan_verify_skip_gpu.json",
            "--submit-path",
            "artifacts/ci/slurm_submit_plan_dry_run.json",
            "--submit-path",
            "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json",
            "--required-baseline-scenario",
            "configs/scenarios/s1_ann_frontier_d3.yaml",
            "--report-input",
            "artifacts/figures/ci_preflight",
            "--require-report-policy",
            "--output",
            "artifacts/ci/ci_protocol_audit.json",
            "--strict",
            "--json",
        ]
    )
    assert code == 53
    assert captured["argv"] == [
        "--config-dir",
        "configs/scenarios",
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--manifest-dir",
        "maxionbench/datasets/manifests",
        "--required-baseline-scenario",
        "configs/scenarios/s1_ann_frontier_d3.yaml",
        "--output",
        "artifacts/ci/ci_protocol_audit.json",
        "--verify-path",
        "artifacts/ci/slurm_plan_verify.json",
        "--verify-path",
        "artifacts/ci/slurm_plan_verify_skip_gpu.json",
        "--submit-path",
        "artifacts/ci/slurm_submit_plan_dry_run.json",
        "--submit-path",
        "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json",
        "--report-input",
        "artifacts/figures/ci_preflight",
        "--require-report-policy",
        "--strict",
        "--json",
    ]


def test_cli_ci_protocol_audit_dispatches_strict_d3_scenario_scale(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 153

    monkeypatch.setattr(ci_protocol_audit_mod, "main", _fake_main)
    code = cli_main(
        [
            "ci-protocol-audit",
            "--config-dir",
            "configs/scenarios",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--manifest-dir",
            "maxionbench/datasets/manifests",
            "--strict-d3-scenario-scale",
            "--json",
        ]
    )
    assert code == 153
    assert captured["argv"] == [
        "--config-dir",
        "configs/scenarios",
        "--slurm-dir",
        "maxionbench/orchestration/slurm",
        "--manifest-dir",
        "maxionbench/datasets/manifests",
        "--required-baseline-scenario",
        "configs/scenarios/s1_ann_frontier_d3.yaml",
        "--output",
        "artifacts/ci/ci_protocol_audit.json",
        "--strict-d3-scenario-scale",
        "--json",
    ]
