from __future__ import annotations

from pathlib import Path

from maxionbench.orchestration.slurm.submit_plan import build_submit_steps, submit_steps


def test_build_submit_steps_enforces_d3_dependency_chain() -> None:
    steps = build_submit_steps(include_gpu=True)
    by_key = {step.key: step for step in steps}
    assert by_key["calibrate"].depends_on == ()
    assert by_key["cpu_d3_baseline"].depends_on == ("calibrate",)
    assert by_key["cpu_d3_workloads"].depends_on == ("calibrate", "cpu_d3_baseline")
    assert by_key["cpu_non_d3"].depends_on == ("calibrate",)
    assert by_key["gpu_all"].depends_on == ("calibrate",)


def test_submit_steps_dry_run_resolves_afterok_dependencies(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        dry_run=True,
    )
    by_key = {step["key"]: step for step in summary["steps"]}

    assert by_key["cpu_d3_baseline"]["command"][2:4] == ["--dependency", "afterok:<CALIBRATE_JOB_ID>"]
    assert by_key["cpu_d3_workloads"]["command"][2:4] == [
        "--dependency",
        "afterok:<CALIBRATE_JOB_ID>:<CPU_D3_BASELINE_JOB_ID>",
    ]
    assert by_key["cpu_non_d3"]["command"][2:4] == ["--dependency", "afterok:<CALIBRATE_JOB_ID>"]
    assert by_key["gpu_all"]["command"][2:4] == ["--dependency", "afterok:<CALIBRATE_JOB_ID>"]
