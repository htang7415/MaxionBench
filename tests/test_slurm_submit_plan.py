from __future__ import annotations

from pathlib import Path

from maxionbench.orchestration.slurm.run_manifest import RunManifest, RunManifestRow
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


def test_build_submit_steps_can_prepend_dataset_prefetch() -> None:
    steps = build_submit_steps(include_gpu=True, prefetch_datasets=True)
    by_key = {step.key: step for step in steps}
    assert steps[0].key == "prefetch_datasets"
    assert by_key["prefetch_datasets"].depends_on == ()
    assert by_key["calibrate"].depends_on == ("prefetch_datasets",)


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


def test_submit_steps_dry_run_resolves_prefetch_dependency_when_enabled(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("prefetch_datasets.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True, prefetch_datasets=True),
        seed=42,
        prefetch_datasets=True,
        dry_run=True,
    )
    by_key = {step["key"]: step for step in summary["steps"]}

    assert by_key["prefetch_datasets"]["command"][-1] == str((slurm_dir / "prefetch_datasets.sh").resolve())
    assert by_key["calibrate"]["command"][2:4] == ["--dependency", "afterok:<PREFETCH_DATASETS_JOB_ID>"]
    assert summary["prefetch_datasets"] is True


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


def test_submit_steps_rejects_unknown_local_slurm_profile(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(tmp_path / "missing_profiles_local.yaml"))

    try:
        submit_steps(
            slurm_dir=slurm_dir,
            steps=build_submit_steps(include_gpu=False),
            seed=42,
            slurm_profile="missing_cluster",
            dry_run=True,
        )
    except ValueError as exc:
        assert "unknown slurm profile" in str(exc)
        assert "profiles_local.yaml" in str(exc)
    else:
        raise AssertionError("expected submit_steps to reject an undefined local slurm profile")


def test_submit_steps_applies_local_profile_override_flags(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    overrides_path = tmp_path / "profiles_local.yaml"
    overrides_path.write_text(
        """
your_cluster:
  base:
    - ["--job-name", "maxion"]
    - ["--output", "logs/%x_%j.out"]
    - ["--error", "logs/%x_%j.err"]
    - ["--nodes", "1"]
    - ["--ntasks-per-node", "1"]
    - ["--cpus-per-task", "64"]
    - ["--mem", "256G"]
    - ["--time", "2-00:00:00"]
    - ["--account", "private-account"]
    - ["--partition", "private-partition"]
  step_overrides:
    calibrate:
      - ["--gres", "gpu:1"]
    gpu_all:
      - ["--gres", "gpu:1"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(overrides_path))

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )
    first_cmd = [str(item) for item in summary["steps"][0]["command"]]
    assert "--job-name" in first_cmd
    assert first_cmd[first_cmd.index("--job-name") + 1] == "maxion"
    assert "--partition" in first_cmd
    assert first_cmd[first_cmd.index("--partition") + 1] == "private-partition"
    assert "--account" in first_cmd
    assert first_cmd[first_cmd.index("--account") + 1] == "private-account"
    assert "--gres" in first_cmd
    assert first_cmd[first_cmd.index("--gres") + 1] == "gpu:1"
    assert summary["slurm_profile"] == "your_cluster"

    cpu_cmd = [str(item) for item in summary["steps"][1]["command"]]
    assert "--partition" in cpu_cmd
    assert "--account" in cpu_cmd
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


def test_submit_steps_exports_container_runtime_settings_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False),
        seed=42,
        container_runtime="apptainer",
        container_image="/shared/containers/maxionbench.sif",
        container_bind=["/shared/datasets", "/shared/models/hf:/shared/models/hf"],
        hf_cache_dir="/shared/models/hf",
        dry_run=True,
    )
    assert summary["container_runtime"] == "apptainer"
    assert summary["container_image"] == "/shared/containers/maxionbench.sif"
    assert summary["container_bind"] == ["/shared/datasets", "/shared/models/hf:/shared/models/hf"]
    assert summary["hf_cache_dir"] == "/shared/models/hf"
    for step in summary["steps"]:
        cmd = [str(item) for item in step["command"]]
        assert "--export" in cmd
        export_value = cmd[cmd.index("--export") + 1]
        assert "MAXIONBENCH_CONTAINER_RUNTIME=apptainer" in export_value
        assert "MAXIONBENCH_CONTAINER_IMAGE=/shared/containers/maxionbench.sif" in export_value
        assert "MAXIONBENCH_CONTAINER_BIND=/shared/datasets|/shared/models/hf:/shared/models/hf" in export_value
        assert "MAXIONBENCH_HF_CACHE_DIR=/shared/models/hf" in export_value


def test_submit_steps_requires_container_image_when_runtime_enabled(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    try:
        submit_steps(
            slurm_dir=slurm_dir,
            steps=build_submit_steps(include_gpu=False),
            seed=42,
            container_runtime="apptainer",
            dry_run=True,
        )
    except ValueError as exc:
        assert "container_image is required" in str(exc)
    else:
        raise AssertionError("expected submit_steps to reject missing container_image")


def test_build_submit_steps_supports_manifest_dataset_pipeline_and_postprocess() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/artifacts/slurm_manifests/latest/generated_configs",
        cpu_rows=[
            RunManifestRow(
                group="cpu",
                config_path="/repo/generated/s1_d3.yaml",
                engine="qdrant",
                scenario="s1_ann_frontier",
                dataset_bundle="D3",
                template_name="s1_ann_frontier_d3.yaml",
            ),
            RunManifestRow(
                group="cpu",
                config_path="/repo/generated/s3.yaml",
                engine="qdrant",
                scenario="s3_churn_smooth",
                dataset_bundle="D3",
                template_name="s3_churn_smooth.yaml",
            ),
            RunManifestRow(
                group="cpu",
                config_path="/repo/generated/s4.yaml",
                engine="qdrant",
                scenario="s4_hybrid",
                dataset_bundle="D4",
                template_name="s4_hybrid.yaml",
            ),
        ],
        gpu_rows=[
            RunManifestRow(
                group="gpu",
                config_path="/repo/generated/s5.yaml",
                engine="faiss-gpu",
                scenario="s5_rerank",
                dataset_bundle="D4",
                template_name="s5_rerank.yaml",
            )
        ],
        selected_engines=["qdrant", "faiss-gpu"],
        selected_templates=["s1_ann_frontier_d3.yaml", "s3_churn_smooth.yaml", "s4_hybrid.yaml", "s5_rerank.yaml"],
    )

    steps = build_submit_steps(
        include_gpu=True,
        download_datasets=True,
        preprocess_datasets=True,
        include_postprocess=True,
        manifest=manifest,
    )
    by_key = {step.key: step for step in steps}

    assert [step.key for step in steps] == [
        "download_datasets",
        "preprocess_datasets",
        "calibrate",
        "cpu_d3_baseline",
        "cpu_d3_workloads",
        "cpu_non_d3",
        "gpu_all",
        "postprocess",
    ]
    assert by_key["download_datasets"].depends_on == ()
    assert by_key["preprocess_datasets"].depends_on == ("download_datasets",)
    assert by_key["calibrate"].depends_on == ("preprocess_datasets",)
    assert by_key["cpu_d3_baseline"].array == "0"
    assert by_key["cpu_d3_workloads"].array == "1"
    assert by_key["cpu_non_d3"].array == "2"
    assert by_key["gpu_all"].array == "0"
    assert by_key["postprocess"].depends_on == (
        "cpu_d3_baseline",
        "cpu_d3_workloads",
        "cpu_non_d3",
        "gpu_all",
    )


def test_submit_steps_exports_run_manifest_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "download_datasets.sh",
        "preprocess_datasets.sh",
        "calibrate_d3.sh",
        "cpu_array.sh",
        "gpu_array.sh",
        "postprocess.sh",
    ):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(
            include_gpu=True,
            download_datasets=True,
            preprocess_datasets=True,
            include_postprocess=True,
        ),
        seed=42,
        run_manifest="artifacts/slurm_manifests/latest/run_manifest.json",
        dry_run=True,
    )

    for step in summary["steps"]:
        command = [str(item) for item in step["command"]]
        assert "--export" in command
        export_value = command[command.index("--export") + 1]
        assert "MAXIONBENCH_SLURM_RUN_MANIFEST=artifacts/slurm_manifests/latest/run_manifest.json" in export_value


def test_submit_steps_uses_tracked_cluster_profile_examples(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        slurm_profile="euler_apptainer",
        dry_run=True,
    )
    calibrate_cmd = [str(item) for item in summary["steps"][0]["command"]]
    assert "--job-name" in calibrate_cmd
    assert calibrate_cmd[calibrate_cmd.index("--job-name") + 1] == "maxion"
    assert "--cpus-per-task" in calibrate_cmd
    assert calibrate_cmd[calibrate_cmd.index("--cpus-per-task") + 1] == "96"
    assert "--gres" in calibrate_cmd
    assert calibrate_cmd[calibrate_cmd.index("--gres") + 1] == "gpu:1"
