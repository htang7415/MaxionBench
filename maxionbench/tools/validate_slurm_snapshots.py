"""Validate Slurm verification/submission snapshot JSON artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

DEFAULT_VERIFY_PATHS = (
    "artifacts/ci/slurm_plan_verify.json",
    "artifacts/ci/slurm_plan_verify_skip_gpu.json",
)
DEFAULT_SUBMIT_PATHS = (
    "artifacts/ci/slurm_submit_plan_dry_run.json",
    "artifacts/ci/slurm_submit_plan_skip_gpu_dry_run.json",
)
DEFAULT_REQUIRED_STEP_KEYS = ("calibrate", "cpu_d3_baseline", "cpu_d3_workloads", "cpu_non_d3")
DEFAULT_REQUIRED_DEPENDS = {
    "calibrate": [],
    "cpu_d3_baseline": ["calibrate"],
    "cpu_d3_workloads": ["calibrate", "cpu_d3_baseline"],
    "cpu_non_d3": ["calibrate"],
    "gpu_all": ["calibrate"],
}
DEFAULT_REQUIRED_EXPORT_TOKEN = "ALL"
DEFAULT_REQUIRED_EXPORT_SEED_KEY = "MAXIONBENCH_SEED"
DEFAULT_PAPER_SCENARIO_CONFIG_DIR = "configs/scenarios_paper"


def validate_slurm_snapshots(
    *,
    verify_paths: list[Path],
    submit_paths: list[Path],
    required_baseline_scenario: str = "configs/scenarios/s1_ann_frontier_d3.yaml",
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []

    for path in verify_paths:
        payload = _load_json_object(path, errors)
        if payload is None:
            continue
        if payload.get("pass") is not True:
            _err(errors, path, "pass", True, payload.get("pass"))
        if int(payload.get("error_count", -1)) != 0:
            _err(errors, path, "error_count", 0, payload.get("error_count"))
        cpu_scenarios = payload.get("cpu_scenarios")
        if not isinstance(cpu_scenarios, list) or not cpu_scenarios:
            _err(errors, path, "cpu_scenarios", "non-empty list", cpu_scenarios)
        elif required_baseline_scenario not in cpu_scenarios:
            _err(errors, path, "cpu_scenarios", f"contains {required_baseline_scenario!r}", cpu_scenarios)

    for path in submit_paths:
        payload = _load_json_object(path, errors)
        if payload is None:
            continue
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            _err(errors, path, "steps", "non-empty list", steps)
            continue

        keys = {str(item.get("key")) for item in steps if isinstance(item, dict)}
        step_by_key = {str(item.get("key")): item for item in steps if isinstance(item, dict)}
        missing = sorted(set(DEFAULT_REQUIRED_STEP_KEYS) - keys)
        if missing:
            _err(errors, path, "required_step_keys", list(DEFAULT_REQUIRED_STEP_KEYS), sorted(keys))

        is_skip_gpu = "skip_gpu" in path.name
        is_paper_snapshot = "paper" in path.name
        if is_skip_gpu and "gpu_all" in keys:
            _err(errors, path, "gpu_all", "absent in skip-gpu mode", "present")
        if not is_skip_gpu and "gpu_all" not in keys:
            _err(errors, path, "gpu_all", "present in default mode", "absent")

        expected_depends = dict(DEFAULT_REQUIRED_DEPENDS)
        if is_skip_gpu:
            expected_depends.pop("gpu_all", None)
        for step_key, expected in expected_depends.items():
            step = step_by_key.get(step_key)
            if not isinstance(step, dict):
                continue
            depends_on = step.get("depends_on")
            if not isinstance(depends_on, list):
                _err(errors, path, f"{step_key}.depends_on", expected, depends_on)
                continue
            actual_depends = [str(item) for item in depends_on]
            if actual_depends != expected:
                _err(errors, path, f"{step_key}.depends_on", expected, actual_depends)

            command = step.get("command")
            dependency_flag_value = _command_flag_value(command, "--dependency")
            dependencies_resolved = step.get("dependencies_resolved")
            if not isinstance(dependencies_resolved, list):
                _err(errors, path, f"{step_key}.dependencies_resolved", "list", dependencies_resolved)
                continue
            resolved = [str(item) for item in dependencies_resolved]
            if len(resolved) != len(expected):
                _err(errors, path, f"{step_key}.dependencies_resolved.length", len(expected), len(resolved))
            if expected:
                if not dependency_flag_value:
                    _err(
                        errors,
                        path,
                        f"{step_key}.command.--dependency",
                        "afterok:<job_ids>",
                        dependency_flag_value,
                    )
                else:
                    expected_dependency = f"afterok:{':'.join(resolved)}"
                    if dependency_flag_value != expected_dependency:
                        _err(
                            errors,
                            path,
                            f"{step_key}.command.--dependency",
                            expected_dependency,
                            dependency_flag_value,
                        )
            elif dependency_flag_value is not None:
                _err(errors, path, f"{step_key}.command.--dependency", None, dependency_flag_value)

            export_flag_value = _command_flag_value(command, "--export")
            if not export_flag_value:
                _err(errors, path, f"{step_key}.command.--export", "non-empty export payload", export_flag_value)
                continue
            export_tokens, export_kv = _parse_export_payload(export_flag_value)
            if DEFAULT_REQUIRED_EXPORT_TOKEN not in export_tokens:
                _err(
                    errors,
                    path,
                    f"{step_key}.command.--export.{DEFAULT_REQUIRED_EXPORT_TOKEN}",
                    "present",
                    "absent",
                )
            if DEFAULT_REQUIRED_EXPORT_SEED_KEY not in export_kv:
                _err(
                    errors,
                    path,
                    f"{step_key}.command.--export.{DEFAULT_REQUIRED_EXPORT_SEED_KEY}",
                    "present",
                    "absent",
                )
            if is_paper_snapshot:
                actual_scenario_dir = export_kv.get("MAXIONBENCH_SCENARIO_CONFIG_DIR")
                if actual_scenario_dir != DEFAULT_PAPER_SCENARIO_CONFIG_DIR:
                    _err(
                        errors,
                        path,
                        f"{step_key}.command.--export.MAXIONBENCH_SCENARIO_CONFIG_DIR",
                        DEFAULT_PAPER_SCENARIO_CONFIG_DIR,
                        actual_scenario_dir,
                    )

    return {
        "pass": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "verify_paths": [str(path) for path in verify_paths],
        "submit_paths": [str(path) for path in submit_paths],
        "required_baseline_scenario": required_baseline_scenario,
    }


def _load_json_object(path: Path, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    resolved = path.resolve()
    if not resolved.exists():
        _err(errors, resolved, "exists", True, False)
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        _err(errors, resolved, "json", "valid object", f"decode_error: {exc}")
        return None
    if not isinstance(payload, dict):
        _err(errors, resolved, "json_type", "object", type(payload).__name__)
        return None
    return payload


def _err(errors: list[dict[str, Any]], path: Path, field: str, expected: Any, actual: Any) -> None:
    errors.append(
        {
            "file": str(path),
            "field": field,
            "expected": expected,
            "actual": actual,
            "message": f"{path}: `{field}` expected {expected!r}, got {actual!r}",
        }
    )


def _command_flag_value(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    values = [str(item) for item in command]
    for idx, token in enumerate(values):
        if token != flag:
            continue
        nxt = idx + 1
        if nxt >= len(values):
            return ""
        return values[nxt]
    return None


def _parse_export_payload(value: str) -> tuple[set[str], dict[str, str]]:
    tokens: set[str] = set()
    kv: dict[str, str] = {}
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        tokens.add(token)
        if "=" not in token:
            continue
        key, val = token.split("=", maxsplit=1)
        kv[key.strip()] = val.strip()
    return tokens, kv


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate Slurm snapshot artifacts emitted by preflight CI")
    parser.add_argument(
        "--verify-path",
        action="append",
        dest="verify_paths",
        default=None,
        help="Path to a verify-slurm-plan JSON artifact (repeatable)",
    )
    parser.add_argument(
        "--submit-path",
        action="append",
        dest="submit_paths",
        default=None,
        help="Path to a submit-slurm-plan dry-run JSON artifact (repeatable)",
    )
    parser.add_argument(
        "--required-baseline-scenario",
        default="configs/scenarios/s1_ann_frontier_d3.yaml",
        help="Scenario entry that must appear in verify snapshot cpu_scenarios",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    verify_paths = [Path(item) for item in (args.verify_paths or list(DEFAULT_VERIFY_PATHS))]
    submit_paths = [Path(item) for item in (args.submit_paths or list(DEFAULT_SUBMIT_PATHS))]
    summary = validate_slurm_snapshots(
        verify_paths=verify_paths,
        submit_paths=submit_paths,
        required_baseline_scenario=str(args.required_baseline_scenario),
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print(
                "slurm snapshot validation passed: "
                f"{len(verify_paths)} verify files + {len(submit_paths)} submit files"
            )
        else:
            print(f"slurm snapshot validation failed: {summary['error_count']} issue(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
