from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxionbench.tools import ci_protocol_audit as audit_mod


def _patch_core_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pins_pass: bool = True,
    pins_error_count: int = 0,
    manifests_pass: bool = True,
    manifests_error_count: int = 0,
    slurm_default_pass: bool = True,
    slurm_default_error_count: int = 0,
    slurm_skip_gpu_pass: bool = True,
    slurm_skip_gpu_error_count: int = 0,
    snapshot_pass: bool = True,
    snapshot_error_count: int = 0,
) -> None:
    def _fake_verify_scenario_config_dir(_: Path) -> dict[str, object]:
        return {"pass": pins_pass, "error_count": pins_error_count}

    def _fake_verify_dataset_manifest_dir(_: Path) -> dict[str, object]:
        return {"pass": manifests_pass, "error_count": manifests_error_count}

    def _fake_verify_slurm_plan(*, slurm_dir: Path, include_gpu: bool) -> dict[str, object]:
        assert isinstance(slurm_dir, Path)
        if include_gpu:
            return {"pass": slurm_default_pass, "error_count": slurm_default_error_count}
        return {"pass": slurm_skip_gpu_pass, "error_count": slurm_skip_gpu_error_count}

    def _fake_validate_slurm_snapshots(
        *,
        verify_paths: list[Path],
        submit_paths: list[Path],
        required_baseline_scenario: str,
    ) -> dict[str, object]:
        assert verify_paths
        assert submit_paths
        assert required_baseline_scenario
        return {"pass": snapshot_pass, "error_count": snapshot_error_count}

    monkeypatch.setattr(audit_mod, "verify_scenario_config_dir", _fake_verify_scenario_config_dir)
    monkeypatch.setattr(audit_mod, "verify_dataset_manifest_dir", _fake_verify_dataset_manifest_dir)
    monkeypatch.setattr(audit_mod, "verify_slurm_plan", _fake_verify_slurm_plan)
    monkeypatch.setattr(audit_mod, "validate_slurm_snapshots", _fake_validate_slurm_snapshots)


def test_run_ci_protocol_audit_passes_when_required_checks_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_core_checks(monkeypatch)

    summary = audit_mod.run_ci_protocol_audit(
        config_dir=tmp_path / "configs",
        slurm_dir=tmp_path / "slurm",
        manifest_dir=tmp_path / "manifests",
        verify_paths=[tmp_path / "verify_a.json", tmp_path / "verify_b.json"],
        submit_paths=[tmp_path / "submit_a.json", tmp_path / "submit_b.json"],
        required_baseline_scenario="configs/scenarios/s1_ann_frontier_d3.yaml",
        report_input=None,
        require_report_policy=False,
    )

    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["checks"]["verify_dataset_manifests"]["pass"] is True
    assert summary["checks"]["report_output_policy"]["skipped"] is True
    assert summary["checks"]["report_output_policy"]["pass"] is None


def test_run_ci_protocol_audit_ignores_report_policy_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_core_checks(monkeypatch)
    monkeypatch.setattr(audit_mod, "inspect_report_output_policy", lambda _: {"pass": False, "error_count": 4})

    report_input = tmp_path / "figures"
    report_input.mkdir(parents=True, exist_ok=True)
    summary = audit_mod.run_ci_protocol_audit(
        config_dir=tmp_path / "configs",
        slurm_dir=tmp_path / "slurm",
        manifest_dir=tmp_path / "manifests",
        verify_paths=[tmp_path / "verify_a.json", tmp_path / "verify_b.json"],
        submit_paths=[tmp_path / "submit_a.json", tmp_path / "submit_b.json"],
        required_baseline_scenario="configs/scenarios/s1_ann_frontier_d3.yaml",
        report_input=report_input,
        require_report_policy=False,
    )

    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["checks"]["verify_dataset_manifests"]["pass"] is True
    assert summary["checks"]["report_output_policy"]["pass"] is False
    assert int(summary["checks"]["report_output_policy"]["error_count"]) == 4


def test_ci_protocol_audit_main_strict_fails_when_required_report_policy_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_core_checks(monkeypatch)
    monkeypatch.setattr(audit_mod, "inspect_report_output_policy", lambda _: {"pass": False, "error_count": 2})

    output = tmp_path / "ci_protocol_audit.json"
    report_input = tmp_path / "figures"
    report_input.mkdir(parents=True, exist_ok=True)
    code = audit_mod.main(
        [
            "--config-dir",
            str(tmp_path / "configs"),
            "--slurm-dir",
            str(tmp_path / "slurm"),
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--verify-path",
            str(tmp_path / "verify_a.json"),
            "--verify-path",
            str(tmp_path / "verify_b.json"),
            "--submit-path",
            str(tmp_path / "submit_a.json"),
            "--submit-path",
            str(tmp_path / "submit_b.json"),
            "--required-baseline-scenario",
            "configs/scenarios/s1_ann_frontier_d3.yaml",
            "--report-input",
            str(report_input),
            "--require-report-policy",
            "--output",
            str(output),
            "--strict",
            "--json",
        ]
    )

    assert code == 2
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pass"] is False
    assert int(payload["error_count"]) == 2
    assert payload["checks"]["report_output_policy"]["pass"] is False
    assert int(payload["checks"]["report_output_policy"]["error_count"]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["pass"] is False


def test_run_ci_protocol_audit_fails_when_dataset_manifests_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_core_checks(monkeypatch, manifests_pass=False, manifests_error_count=3)
    summary = audit_mod.run_ci_protocol_audit(
        config_dir=tmp_path / "configs",
        slurm_dir=tmp_path / "slurm",
        manifest_dir=tmp_path / "manifests",
        verify_paths=[tmp_path / "verify_a.json", tmp_path / "verify_b.json"],
        submit_paths=[tmp_path / "submit_a.json", tmp_path / "submit_b.json"],
        required_baseline_scenario="configs/scenarios/s1_ann_frontier_d3.yaml",
        report_input=None,
        require_report_policy=False,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) == 3
    assert summary["checks"]["verify_dataset_manifests"]["pass"] is False
    assert int(summary["checks"]["verify_dataset_manifests"]["error_count"]) == 3
