"""Consolidated CI protocol audit for pinned benchmark policies."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.tools.report_output_policy import inspect_report_output_policy
from maxionbench.tools.verify_dataset_manifests import verify_dataset_manifest_dir
from maxionbench.tools.validate_slurm_snapshots import (
    DEFAULT_SUBMIT_PATHS,
    DEFAULT_VERIFY_PATHS,
    validate_slurm_snapshots,
)
from maxionbench.tools.verify_pins import verify_scenario_config_dir
from maxionbench.tools.verify_slurm_plan import verify_slurm_plan


def run_ci_protocol_audit(
    *,
    config_dir: Path,
    slurm_dir: Path,
    manifest_dir: Path,
    verify_paths: list[Path],
    submit_paths: list[Path],
    required_baseline_scenario: str,
    report_input: Path | None = None,
    require_report_policy: bool = False,
    strict_d3_scenario_scale: bool = False,
) -> dict[str, Any]:
    pins = verify_scenario_config_dir(
        config_dir,
        strict_d3_scenario_scale=bool(strict_d3_scenario_scale),
    )
    manifests = verify_dataset_manifest_dir(manifest_dir)
    slurm_default = verify_slurm_plan(slurm_dir=slurm_dir, include_gpu=True)
    slurm_skip_gpu = verify_slurm_plan(slurm_dir=slurm_dir, include_gpu=False)
    slurm_snapshots = validate_slurm_snapshots(
        verify_paths=verify_paths,
        submit_paths=submit_paths,
        required_baseline_scenario=required_baseline_scenario,
    )

    report_policy: dict[str, Any]
    if report_input is None:
        report_policy = {
            "pass": None,
            "skipped": True,
            "reason": "report_input not provided",
            "error_count": 0,
        }
    else:
        payload = inspect_report_output_policy(report_input)
        report_policy = {
            **payload,
            "skipped": False,
        }

    checks = {
        "verify_pins": {
            "pass": bool(pins.get("pass", False)),
            "error_count": int(pins.get("error_count", 0)),
        },
        "verify_dataset_manifests": {
            "pass": bool(manifests.get("pass", False)),
            "error_count": int(manifests.get("error_count", 0)),
        },
        "verify_slurm_plan_default": {
            "pass": bool(slurm_default.get("pass", False)),
            "error_count": int(slurm_default.get("error_count", 0)),
        },
        "verify_slurm_plan_skip_gpu": {
            "pass": bool(slurm_skip_gpu.get("pass", False)),
            "error_count": int(slurm_skip_gpu.get("error_count", 0)),
        },
        "validate_slurm_snapshots": {
            "pass": bool(slurm_snapshots.get("pass", False)),
            "error_count": int(slurm_snapshots.get("error_count", 0)),
        },
        "report_output_policy": {
            "pass": report_policy.get("pass"),
            "error_count": int(report_policy.get("error_count", 0)),
            "skipped": bool(report_policy.get("skipped", False)),
        },
    }

    required_passes = [
        checks["verify_pins"]["pass"],
        checks["verify_dataset_manifests"]["pass"],
        checks["verify_slurm_plan_default"]["pass"],
        checks["verify_slurm_plan_skip_gpu"]["pass"],
        checks["validate_slurm_snapshots"]["pass"],
    ]
    if require_report_policy:
        required_passes.append(bool(report_policy.get("pass", False)))
    overall_pass = all(required_passes)
    overall_error_count = (
        checks["verify_pins"]["error_count"]
        + checks["verify_dataset_manifests"]["error_count"]
        + checks["verify_slurm_plan_default"]["error_count"]
        + checks["verify_slurm_plan_skip_gpu"]["error_count"]
        + checks["validate_slurm_snapshots"]["error_count"]
        + (checks["report_output_policy"]["error_count"] if require_report_policy else 0)
    )

    return {
        "pass": overall_pass,
        "error_count": int(overall_error_count),
        "require_report_policy": bool(require_report_policy),
        "inputs": {
            "config_dir": str(config_dir.resolve()),
            "slurm_dir": str(slurm_dir.resolve()),
            "manifest_dir": str(manifest_dir.resolve()),
            "strict_d3_scenario_scale": bool(strict_d3_scenario_scale),
            "verify_paths": [str(path.resolve()) for path in verify_paths],
            "submit_paths": [str(path.resolve()) for path in submit_paths],
            "required_baseline_scenario": required_baseline_scenario,
            "report_input": str(report_input.resolve()) if report_input else None,
        },
        "checks": checks,
        "details": {
            "verify_pins": pins,
            "verify_dataset_manifests": manifests,
            "verify_slurm_plan_default": slurm_default,
            "verify_slurm_plan_skip_gpu": slurm_skip_gpu,
            "validate_slurm_snapshots": slurm_snapshots,
            "report_output_policy": report_policy,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run consolidated CI protocol audit checks")
    parser.add_argument("--config-dir", default="configs/scenarios")
    parser.add_argument("--slurm-dir", default="maxionbench/orchestration/slurm")
    parser.add_argument("--manifest-dir", default="maxionbench/datasets/manifests")
    parser.add_argument("--verify-path", action="append", dest="verify_paths", default=None)
    parser.add_argument("--submit-path", action="append", dest="submit_paths", default=None)
    parser.add_argument(
        "--required-baseline-scenario",
        default="configs/scenarios/s1_ann_frontier_d3.yaml",
    )
    parser.add_argument("--report-input", default=None)
    parser.add_argument("--require-report-policy", action="store_true")
    parser.add_argument("--strict-d3-scenario-scale", action="store_true")
    parser.add_argument("--output", default="artifacts/ci/ci_protocol_audit.json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    verify_paths = [Path(item) for item in (args.verify_paths or list(DEFAULT_VERIFY_PATHS))]
    submit_paths = [Path(item) for item in (args.submit_paths or list(DEFAULT_SUBMIT_PATHS))]
    report_input = Path(args.report_input) if args.report_input else None
    summary = run_ci_protocol_audit(
        config_dir=Path(args.config_dir),
        slurm_dir=Path(args.slurm_dir),
        manifest_dir=Path(args.manifest_dir),
        verify_paths=verify_paths,
        submit_paths=submit_paths,
        required_baseline_scenario=str(args.required_baseline_scenario),
        report_input=report_input,
        require_report_policy=bool(args.require_report_policy),
        strict_d3_scenario_scale=bool(args.strict_d3_scenario_scale),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        state = "pass" if summary["pass"] else "fail"
        print(f"{state}: ci protocol audit error_count={summary['error_count']}")

    if args.strict and not bool(summary["pass"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
