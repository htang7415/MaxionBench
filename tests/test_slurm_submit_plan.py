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


def test_build_submit_steps_supports_skip_s6_deferral() -> None:
    steps = build_submit_steps(include_gpu=True, skip_s6=True)
    by_key = {step.key: step for step in steps}
    assert by_key["cpu_non_d3"].array == "0,5"
    assert by_key["gpu_all"].array == "0-2"


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


def test_submit_steps_exports_scenario_config_dir_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False),
        seed=42,
        scenario_config_dir="configs/scenarios_paper",
        dry_run=True,
    )
    assert summary["scenario_config_dir"] == "configs/scenarios_paper"
    for step in summary["steps"]:
        command = [str(item) for item in step["command"]]
        assert "--export" in command
        export_value = command[command.index("--export") + 1]
        assert "MAXIONBENCH_SEED=42" in export_value
        assert "MAXIONBENCH_SCENARIO_CONFIG_DIR=configs/scenarios_paper" in export_value


def test_submit_steps_applies_named_slurm_profile_flags(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False),
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )
    first_cmd = [str(item) for item in summary["steps"][0]["command"]]
    assert "--job-name" in first_cmd
    assert first_cmd[first_cmd.index("--job-name") + 1] == "maxion"
    assert "--partition" in first_cmd
    assert first_cmd[first_cmd.index("--partition") + 1] == "pdelab"
    assert "--gres" in first_cmd
    assert first_cmd[first_cmd.index("--gres") + 1] == "gpu:1"
    assert summary["slurm_profile"] == "your_cluster"

    cpu_cmd = [str(item) for item in summary["steps"][1]["command"]]
    assert "--partition" in cpu_cmd
    assert cpu_cmd[cpu_cmd.index("--partition") + 1] == "pdelab"
    assert "--gres" not in cpu_cmd


def test_submit_steps_exports_output_root_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False),
        seed=42,
        output_root="artifacts/workstation_runs/example/results/slurm",
        dry_run=True,
    )
    for step in summary["steps"]:
        cmd = [str(item) for item in step["command"]]
        assert "--export" in cmd
        export_value = cmd[cmd.index("--export") + 1]
        assert "MAXIONBENCH_OUTPUT_ROOT=artifacts/workstation_runs/example/results/slurm" in export_value
