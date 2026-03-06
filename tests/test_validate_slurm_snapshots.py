from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maxionbench.tools.validate_slurm_snapshots import validate_slurm_snapshots


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_submit_steps(*, include_gpu: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "key": "calibrate",
            "depends_on": [],
            "dependencies_resolved": [],
            "command": ["sbatch", "--parsable", "calibrate_d3.sh"],
        },
        {
            "key": "cpu_d3_baseline",
            "depends_on": ["calibrate"],
            "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
            "command": ["sbatch", "--parsable", "--dependency", "afterok:<CALIBRATE_JOB_ID>", "cpu_array.sh"],
        },
        {
            "key": "cpu_d3_workloads",
            "depends_on": ["calibrate", "cpu_d3_baseline"],
            "dependencies_resolved": ["<CALIBRATE_JOB_ID>", "<CPU_D3_BASELINE_JOB_ID>"],
            "command": [
                "sbatch",
                "--parsable",
                "--dependency",
                "afterok:<CALIBRATE_JOB_ID>:<CPU_D3_BASELINE_JOB_ID>",
                "cpu_array.sh",
            ],
        },
        {
            "key": "cpu_non_d3",
            "depends_on": ["calibrate"],
            "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
            "command": ["sbatch", "--parsable", "--dependency", "afterok:<CALIBRATE_JOB_ID>", "cpu_array.sh"],
        },
    ]
    if include_gpu:
        steps.append(
            {
                "key": "gpu_all",
                "depends_on": ["calibrate"],
                "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
                "command": ["sbatch", "--parsable", "--dependency", "afterok:<CALIBRATE_JOB_ID>", "gpu_array.sh"],
            }
        )
    return steps


def test_validate_slurm_snapshots_passes_for_valid_payloads(tmp_path: Path) -> None:
    verify_a = tmp_path / "slurm_plan_verify.json"
    verify_b = tmp_path / "slurm_plan_verify_skip_gpu.json"
    submit_a = tmp_path / "slurm_submit_plan_dry_run.json"
    submit_b = tmp_path / "slurm_submit_plan_skip_gpu_dry_run.json"

    verify_payload = {
        "pass": True,
        "error_count": 0,
        "cpu_scenarios": [
            "configs/scenarios/s1_ann_frontier.yaml",
            "configs/scenarios/s1_ann_frontier_d3.yaml",
        ],
    }
    submit_payload = {"steps": _valid_submit_steps(include_gpu=True)}
    submit_skip_gpu_payload = {"steps": _valid_submit_steps(include_gpu=False)}

    _write_json(verify_a, verify_payload)
    _write_json(verify_b, verify_payload)
    _write_json(submit_a, submit_payload)
    _write_json(submit_b, submit_skip_gpu_payload)

    summary = validate_slurm_snapshots(
        verify_paths=[verify_a, verify_b],
        submit_paths=[submit_a, submit_b],
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_validate_slurm_snapshots_fails_on_missing_required_keys(tmp_path: Path) -> None:
    verify_a = tmp_path / "slurm_plan_verify.json"
    verify_b = tmp_path / "slurm_plan_verify_skip_gpu.json"
    submit_a = tmp_path / "slurm_submit_plan_dry_run.json"
    submit_b = tmp_path / "slurm_submit_plan_skip_gpu_dry_run.json"

    _write_json(
        verify_a,
        {
            "pass": True,
            "error_count": 0,
            "cpu_scenarios": ["configs/scenarios/s1_ann_frontier.yaml"],
        },
    )
    _write_json(
        verify_b,
        {
            "pass": False,
            "error_count": 1,
            "cpu_scenarios": [],
        },
    )
    _write_json(
        submit_a,
        {
            "steps": [
                {"key": "calibrate"},
                {"key": "cpu_non_d3"},
            ]
        },
    )
    _write_json(
        submit_b,
        {
            "steps": _valid_submit_steps(include_gpu=True),
        },
    )

    summary = validate_slurm_snapshots(
        verify_paths=[verify_a, verify_b],
        submit_paths=[submit_a, submit_b],
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) > 0
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("cpu_scenarios" in msg for msg in messages)
    assert any("required_step_keys" in msg for msg in messages)
    assert any("gpu_all" in msg for msg in messages)


def test_validate_slurm_snapshots_fails_on_dependency_topology_drift(tmp_path: Path) -> None:
    verify_a = tmp_path / "slurm_plan_verify.json"
    verify_b = tmp_path / "slurm_plan_verify_skip_gpu.json"
    submit_a = tmp_path / "slurm_submit_plan_dry_run.json"
    submit_b = tmp_path / "slurm_submit_plan_skip_gpu_dry_run.json"

    verify_payload = {
        "pass": True,
        "error_count": 0,
        "cpu_scenarios": [
            "configs/scenarios/s1_ann_frontier.yaml",
            "configs/scenarios/s1_ann_frontier_d3.yaml",
        ],
    }
    default_steps = _valid_submit_steps(include_gpu=True)
    for step in default_steps:
        if step["key"] == "cpu_d3_workloads":
            step["depends_on"] = ["calibrate"]
            step["dependencies_resolved"] = ["<CALIBRATE_JOB_ID>"]
            step["command"] = ["sbatch", "--parsable", "--dependency", "afterok:<CALIBRATE_JOB_ID>", "cpu_array.sh"]
            break

    _write_json(verify_a, verify_payload)
    _write_json(verify_b, verify_payload)
    _write_json(submit_a, {"steps": default_steps})
    _write_json(submit_b, {"steps": _valid_submit_steps(include_gpu=False)})

    summary = validate_slurm_snapshots(
        verify_paths=[verify_a, verify_b],
        submit_paths=[submit_a, submit_b],
    )
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("cpu_d3_workloads.depends_on" in msg for msg in messages)
