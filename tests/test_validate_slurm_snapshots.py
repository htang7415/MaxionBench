from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maxionbench.tools.validate_slurm_snapshots import validate_slurm_snapshots


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_submit_steps(
    *,
    include_gpu: bool,
    seed: int = 42,
    scenario_config_dir: str | None = None,
    prefetch_datasets: bool = False,
) -> list[dict[str, Any]]:
    export_parts = ["ALL", f"MAXIONBENCH_SEED={int(seed)}"]
    if scenario_config_dir:
        export_parts.append(f"MAXIONBENCH_SCENARIO_CONFIG_DIR={scenario_config_dir}")
    export_value = ",".join(export_parts)
    steps: list[dict[str, Any]] = []
    if prefetch_datasets:
        steps.append(
            {
                "key": "prefetch_datasets",
                "depends_on": [],
                "dependencies_resolved": [],
                "command": ["sbatch", "--parsable", "--export", export_value, "prefetch_datasets.sh"],
            }
        )
    calibrate_depends = ["prefetch_datasets"] if prefetch_datasets else []
    calibrate_resolved = ["<PREFETCH_DATASETS_JOB_ID>"] if prefetch_datasets else []
    calibrate_command = ["sbatch", "--parsable"]
    if prefetch_datasets:
        calibrate_command.extend(["--dependency", "afterok:<PREFETCH_DATASETS_JOB_ID>"])
    calibrate_command.extend(["--export", export_value, "calibrate_d3.sh"])
    steps.extend(
        [
        {
            "key": "calibrate",
            "depends_on": calibrate_depends,
            "dependencies_resolved": calibrate_resolved,
            "command": calibrate_command,
        },
        {
            "key": "cpu_d3_baseline",
            "depends_on": ["calibrate"],
            "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
            "command": [
                "sbatch",
                "--parsable",
                "--dependency",
                "afterok:<CALIBRATE_JOB_ID>",
                "--export",
                export_value,
                "cpu_array.sh",
            ],
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
                "--export",
                export_value,
                "cpu_array.sh",
            ],
        },
        {
            "key": "cpu_non_d3",
            "depends_on": ["calibrate"],
            "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
            "command": [
                "sbatch",
                "--parsable",
                "--dependency",
                "afterok:<CALIBRATE_JOB_ID>",
                "--export",
                export_value,
                "cpu_array.sh",
            ],
        },
        ]
    )
    if include_gpu:
        steps.append(
            {
                "key": "gpu_all",
                "depends_on": ["calibrate"],
                "dependencies_resolved": ["<CALIBRATE_JOB_ID>"],
                "command": [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    "afterok:<CALIBRATE_JOB_ID>",
                    "--export",
                    export_value,
                    "gpu_array.sh",
                ],
            }
        )
    return steps


def test_validate_slurm_snapshots_passes_for_valid_payloads(tmp_path: Path) -> None:
    verify_a = tmp_path / "slurm_plan_verify.json"
    verify_b = tmp_path / "slurm_plan_verify_skip_gpu.json"
    submit_a = tmp_path / "slurm_submit_plan_dry_run.json"
    submit_b = tmp_path / "slurm_submit_plan_skip_gpu_dry_run.json"
    submit_c = tmp_path / "slurm_submit_plan_paper_skip_gpu_dry_run.json"

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
    submit_paper_skip_gpu_payload = {
        "steps": _valid_submit_steps(include_gpu=False, scenario_config_dir="configs/scenarios_paper")
    }

    _write_json(verify_a, verify_payload)
    _write_json(verify_b, verify_payload)
    _write_json(submit_a, submit_payload)
    _write_json(submit_b, submit_skip_gpu_payload)
    _write_json(submit_c, submit_paper_skip_gpu_payload)

    summary = validate_slurm_snapshots(
        verify_paths=[verify_a, verify_b],
        submit_paths=[submit_a, submit_b, submit_c],
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_validate_slurm_snapshots_accepts_optional_prefetch_step(tmp_path: Path) -> None:
    verify = tmp_path / "slurm_plan_verify.json"
    submit = tmp_path / "slurm_submit_plan_dry_run.json"
    _write_json(
        verify,
        {
            "pass": True,
            "error_count": 0,
            "cpu_scenarios": [
                "configs/scenarios/s1_ann_frontier.yaml",
                "configs/scenarios/s1_ann_frontier_d3.yaml",
            ],
        },
    )
    _write_json(
        submit,
        {
            "steps": _valid_submit_steps(include_gpu=True, prefetch_datasets=True),
        },
    )

    summary = validate_slurm_snapshots(
        verify_paths=[verify],
        submit_paths=[submit],
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
            step["command"] = [
                "sbatch",
                "--parsable",
                "--dependency",
                "afterok:<CALIBRATE_JOB_ID>",
                "--export",
                "ALL,MAXIONBENCH_SEED=42",
                "cpu_array.sh",
            ]
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


def test_validate_slurm_snapshots_fails_when_paper_snapshot_lacks_scenario_override(tmp_path: Path) -> None:
    verify = tmp_path / "slurm_plan_verify.json"
    paper_submit = tmp_path / "slurm_submit_plan_paper_skip_gpu_dry_run.json"
    _write_json(
        verify,
        {
            "pass": True,
            "error_count": 0,
            "cpu_scenarios": [
                "configs/scenarios/s1_ann_frontier.yaml",
                "configs/scenarios/s1_ann_frontier_d3.yaml",
            ],
        },
    )
    _write_json(
        paper_submit,
        {
            "steps": _valid_submit_steps(include_gpu=False),
        },
    )

    summary = validate_slurm_snapshots(
        verify_paths=[verify],
        submit_paths=[paper_submit],
    )
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("MAXIONBENCH_SCENARIO_CONFIG_DIR" in msg for msg in messages)
