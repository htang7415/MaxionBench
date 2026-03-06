"""Verify Slurm submit plan consistency against array scenario layout."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
from typing import Any

from maxionbench.orchestration.slurm.submit_plan import build_submit_steps

D3_BASELINE_SCENARIO = "s1_ann_frontier_d3"
D3_WORKLOAD_SCENARIOS = {"s2_filtered_ann", "s3_churn_smooth", "s3b_churn_bursty"}
GPU_REQUIRED_S5_SCENARIO = "s5_rerank"
GPU_TRACK_B_MARKER = "track_b"
GPU_TRACK_C_MARKER = "track_c"


def verify_slurm_plan(*, slurm_dir: Path, include_gpu: bool = True) -> dict[str, Any]:
    resolved = slurm_dir.resolve()
    errors: list[dict[str, Any]] = []
    gpu_scenarios: list[str] = []

    cpu_array_path = resolved / "cpu_array.sh"
    calibrate_path = resolved / "calibrate_d3.sh"
    gpu_path = resolved / "gpu_array.sh"
    for required in (cpu_array_path, calibrate_path):
        if not required.exists():
            errors.append(
                {
                    "file": str(required),
                    "field": "exists",
                    "expected": True,
                    "actual": False,
                    "message": f"missing required slurm script: {required}",
                }
            )
    if include_gpu and not gpu_path.exists():
        errors.append(
            {
                "file": str(gpu_path),
                "field": "exists",
                "expected": True,
                "actual": False,
                "message": f"missing required slurm script: {gpu_path}",
            }
        )
    if errors:
        return _summary(
            slurm_dir=resolved,
            errors=errors,
            cpu_scenarios=[],
            gpu_scenarios=[],
            include_gpu=include_gpu,
        )

    cpu_scenarios = _extract_cpu_scenarios(cpu_array_path)
    if not cpu_scenarios:
        errors.append(
            {
                "file": str(cpu_array_path),
                "field": "SCENARIOS",
                "expected": "non-empty array",
                "actual": "empty",
                "message": "cpu_array.sh SCENARIOS array is empty or unreadable",
            }
        )
        return _summary(
            slurm_dir=resolved,
            errors=errors,
            cpu_scenarios=cpu_scenarios,
            gpu_scenarios=[],
            include_gpu=include_gpu,
        )

    base_names = [Path(item).stem for item in cpu_scenarios]
    all_indices = set(range(len(cpu_scenarios)))
    try:
        baseline_idx = base_names.index(D3_BASELINE_SCENARIO)
    except ValueError:
        baseline_idx = None
        errors.append(
            {
                "file": str(cpu_array_path),
                "field": D3_BASELINE_SCENARIO,
                "expected": "present in SCENARIOS",
                "actual": "missing",
                "message": f"cpu_array.sh must include {D3_BASELINE_SCENARIO}.yaml for S3/S3b baseline matching",
            }
        )
    d3_workload_indices = {idx for idx, name in enumerate(base_names) if name in D3_WORKLOAD_SCENARIOS}

    expected_non_d3 = set(all_indices)
    if baseline_idx is not None:
        expected_non_d3.discard(baseline_idx)
    expected_non_d3 -= d3_workload_indices

    plan_steps = build_submit_steps(include_gpu=include_gpu)
    by_key = {step.key: step for step in plan_steps}
    required_keys = {"calibrate", "cpu_d3_baseline", "cpu_d3_workloads", "cpu_non_d3"}
    missing = sorted(required_keys - set(by_key.keys()))
    if missing:
        errors.append(
            {
                "file": str((resolved / "submit_plan.py").resolve()),
                "field": "step_keys",
                "expected": sorted(required_keys),
                "actual": sorted(by_key.keys()),
                "message": f"submit plan missing required step keys: {missing}",
            }
        )
        return _summary(
            slurm_dir=resolved,
            errors=errors,
            cpu_scenarios=cpu_scenarios,
            gpu_scenarios=[],
            include_gpu=include_gpu,
        )

    def step_indices(step_key: str) -> set[int]:
        spec = by_key[step_key].array
        if spec is None:
            return set()
        return _parse_array_spec(spec)

    baseline_indices = step_indices("cpu_d3_baseline")
    workload_indices = step_indices("cpu_d3_workloads")
    non_d3_indices = step_indices("cpu_non_d3")

    if baseline_idx is not None:
        _expect_equal(
            errors,
            file=cpu_array_path,
            field="cpu_d3_baseline.array",
            actual=sorted(baseline_indices),
            expected=[baseline_idx],
        )
    _expect_equal(
        errors,
        file=cpu_array_path,
        field="cpu_d3_workloads.array",
        actual=sorted(workload_indices),
        expected=sorted(d3_workload_indices),
    )
    _expect_equal(
        errors,
        file=cpu_array_path,
        field="cpu_non_d3.array",
        actual=sorted(non_d3_indices),
        expected=sorted(expected_non_d3),
    )

    if baseline_indices & workload_indices:
        errors.append(
            {
                "file": str(cpu_array_path),
                "field": "cpu_index_overlap",
                "expected": "disjoint baseline/workload indices",
                "actual": sorted(baseline_indices & workload_indices),
                "message": "cpu_d3_baseline and cpu_d3_workloads arrays must be disjoint",
            }
        )
    covered = baseline_indices | workload_indices | non_d3_indices
    _expect_equal(
        errors,
        file=cpu_array_path,
        field="cpu_index_coverage",
        actual=sorted(covered),
        expected=sorted(all_indices),
    )

    _expect_equal(
        errors,
        file=(resolved / "submit_plan.py").resolve(),
        field="cpu_d3_baseline.depends_on",
        actual=list(by_key["cpu_d3_baseline"].depends_on),
        expected=["calibrate"],
    )
    _expect_equal(
        errors,
        file=(resolved / "submit_plan.py").resolve(),
        field="cpu_d3_workloads.depends_on",
        actual=list(by_key["cpu_d3_workloads"].depends_on),
        expected=["calibrate", "cpu_d3_baseline"],
    )
    _expect_equal(
        errors,
        file=(resolved / "submit_plan.py").resolve(),
        field="cpu_non_d3.depends_on",
        actual=list(by_key["cpu_non_d3"].depends_on),
        expected=["calibrate"],
    )
    if include_gpu:
        gpu_scenarios = _extract_gpu_scenarios(gpu_path)
        if not gpu_scenarios:
            errors.append(
                {
                    "file": str(gpu_path),
                    "field": "SCENARIOS",
                    "expected": "non-empty array",
                    "actual": "empty",
                    "message": "gpu_array.sh SCENARIOS array is empty or unreadable",
                }
            )
        else:
            gpu_base_names = [Path(item).stem for item in gpu_scenarios]
            if GPU_REQUIRED_S5_SCENARIO not in gpu_base_names:
                errors.append(
                    {
                        "file": str(gpu_path),
                        "field": GPU_REQUIRED_S5_SCENARIO,
                        "expected": "present in SCENARIOS",
                        "actual": "missing",
                        "message": "gpu_array.sh must include s5_rerank for pinned S5 GPU workloads",
                    }
                )
            if not any(GPU_TRACK_B_MARKER in name for name in gpu_base_names):
                errors.append(
                    {
                        "file": str(gpu_path),
                        "field": "track_b_marker",
                        "expected": f"scenario name containing `{GPU_TRACK_B_MARKER}`",
                        "actual": gpu_base_names,
                        "message": "gpu_array.sh must include an explicit Track B GPU scenario entry",
                    }
                )
            if not any(GPU_TRACK_C_MARKER in name for name in gpu_base_names):
                errors.append(
                    {
                        "file": str(gpu_path),
                        "field": "track_c_marker",
                        "expected": f"scenario name containing `{GPU_TRACK_C_MARKER}`",
                        "actual": gpu_base_names,
                        "message": "gpu_array.sh must include an explicit Track C GPU scenario entry",
                    }
                )
        if "gpu_all" not in by_key:
            errors.append(
                {
                    "file": str((resolved / "submit_plan.py").resolve()),
                    "field": "gpu_all",
                    "expected": "present",
                    "actual": "missing",
                    "message": "submit plan must include gpu_all step when include_gpu=true",
                }
            )
        else:
            _expect_equal(
                errors,
                file=(resolved / "submit_plan.py").resolve(),
                field="gpu_all.depends_on",
                actual=list(by_key["gpu_all"].depends_on),
                expected=["calibrate"],
            )

    return _summary(
        slurm_dir=resolved,
        errors=errors,
        cpu_scenarios=cpu_scenarios,
        gpu_scenarios=gpu_scenarios,
        include_gpu=include_gpu,
    )


def _extract_cpu_scenarios(cpu_array_path: Path) -> list[str]:
    text = cpu_array_path.read_text(encoding="utf-8")
    match = re.search(r"SCENARIOS=\((.*?)\)", text, flags=re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    return re.findall(r'"([^"]+)"', body)


def _extract_gpu_scenarios(gpu_array_path: Path) -> list[str]:
    text = gpu_array_path.read_text(encoding="utf-8")
    match = re.search(r"SCENARIOS=\((.*?)\)", text, flags=re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    return re.findall(r'"([^"]+)"', body)


def _parse_array_spec(spec: str) -> set[int]:
    values: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", maxsplit=1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid array range `{token}` in spec `{spec}`")
            values.update(range(start, end + 1))
            continue
        values.add(int(token))
    return values


def _expect_equal(
    errors: list[dict[str, Any]],
    *,
    file: Path,
    field: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual == expected:
        return
    errors.append(
        {
            "file": str(file),
            "field": field,
            "expected": expected,
            "actual": actual,
            "message": f"{field} drift: expected {expected!r}, got {actual!r}",
        }
    )


def _summary(
    *,
    slurm_dir: Path,
    errors: list[dict[str, Any]],
    cpu_scenarios: list[str],
    gpu_scenarios: list[str],
    include_gpu: bool,
) -> dict[str, Any]:
    return {
        "slurm_dir": str(slurm_dir),
        "include_gpu": include_gpu,
        "cpu_scenarios": cpu_scenarios,
        "cpu_scenario_count": len(cpu_scenarios),
        "gpu_scenarios": gpu_scenarios,
        "gpu_scenario_count": len(gpu_scenarios),
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify Slurm submit-plan and cpu-array consistency")
    parser.add_argument("--slurm-dir", default="maxionbench/orchestration/slurm")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_slurm_plan(
        slurm_dir=Path(args.slurm_dir),
        include_gpu=not bool(args.skip_gpu),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print(f"slurm plan verification passed: {summary['cpu_scenario_count']} cpu scenarios checked")
        else:
            print(f"slurm plan verification failed: {summary['error_count']} issue(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
