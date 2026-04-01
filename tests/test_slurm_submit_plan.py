from __future__ import annotations

from pathlib import Path

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.slurm.run_manifest import RunManifest, RunManifestRow
from maxionbench.orchestration.slurm.submit_plan import (
    SubmitStep,
    build_submit_steps,
    submit_steps,
    validate_full_matrix_contract,
)


def test_build_submit_steps_enforces_d3_dependency_chain() -> None:
    steps = build_submit_steps(include_gpu=True)
    by_key = {step.key: step for step in steps}
    assert by_key["conformance"].depends_on == ()
    assert by_key["calibrate"].depends_on == ("conformance",)
    assert by_key["cpu_d3_baseline"].depends_on == ("calibrate",)
    assert by_key["cpu_d3_workloads"].depends_on == ("calibrate", "cpu_d3_baseline")
    assert by_key["cpu_non_d3"].depends_on == ("calibrate",)
    assert by_key["gpu_all"].depends_on == ("calibrate",)


def test_build_submit_steps_supports_skip_s6_deferral() -> None:
    steps = build_submit_steps(include_gpu=True, skip_s6=True)
    by_key = {step.key: step for step in steps}
    assert by_key["cpu_non_d3"].array == "0,5"
    assert by_key["gpu_all"].array == "0-2"


def test_build_submit_steps_can_require_prepare_containers_before_conformance() -> None:
    steps = build_submit_steps(include_gpu=True, prepare_containers=True, prefetch_datasets=True)
    by_key = {step.key: step for step in steps}
    assert steps[0].key == "prepare_containers"
    assert by_key["prepare_containers"].depends_on == ()
    assert by_key["prefetch_datasets"].depends_on == ("prepare_containers",)
    assert by_key["conformance"].depends_on == ("prefetch_datasets", "prepare_containers")
    assert by_key["calibrate"].depends_on == ("conformance",)


def test_build_submit_steps_gates_dataset_pipeline_on_prepare_containers() -> None:
    steps = build_submit_steps(
        include_gpu=False,
        prepare_containers=True,
        download_datasets=True,
        preprocess_datasets=True,
    )
    by_key = {step.key: step for step in steps}

    assert steps[0].key == "prepare_containers"
    assert by_key["download_datasets"].depends_on == ("prepare_containers",)
    assert by_key["preprocess_datasets"].depends_on == ("download_datasets", "prepare_containers")
    assert by_key["conformance"].depends_on == ("preprocess_datasets", "prepare_containers")


def test_build_submit_steps_can_prepend_dataset_prefetch() -> None:
    steps = build_submit_steps(include_gpu=True, prefetch_datasets=True)
    by_key = {step.key: step for step in steps}
    assert steps[0].key == "prefetch_datasets"
    assert by_key["prefetch_datasets"].depends_on == ()
    assert by_key["conformance"].depends_on == ("prefetch_datasets",)
    assert by_key["calibrate"].depends_on == ("conformance",)


def test_submit_steps_dry_run_resolves_afterok_dependencies(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        dry_run=True,
    )
    by_key = {step["key"]: step for step in summary["steps"]}

    assert by_key["calibrate"]["command"][2:4] == ["--dependency", "afterok:<CONFORMANCE_JOB_ID>"]
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
    for script_name in ("prefetch_datasets.sh", "conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
    assert by_key["conformance"]["command"][2:4] == ["--dependency", "afterok:<PREFETCH_DATASETS_JOB_ID>"]
    assert by_key["calibrate"]["command"][2:4] == ["--dependency", "afterok:<CONFORMANCE_JOB_ID>"]
    assert summary["prefetch_datasets"] is True


def test_submit_steps_dry_run_resolves_prepare_containers_dependency_when_enabled(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("prepare_containers.sh", "conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True, prepare_containers=True),
        seed=42,
        dry_run=True,
    )
    by_key = {step["key"]: step for step in summary["steps"]}

    assert by_key["prepare_containers"]["command"][-1] == str((slurm_dir / "prepare_containers.sh").resolve())
    assert by_key["conformance"]["command"][2:4] == ["--dependency", "afterok:<PREPARE_CONTAINERS_JOB_ID>"]
    assert by_key["calibrate"]["command"][2:4] == ["--dependency", "afterok:<CONFORMANCE_JOB_ID>"]


def test_submit_steps_dry_run_threads_prepare_containers_through_dataset_pipeline(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "prepare_containers.sh",
        "download_datasets.sh",
        "preprocess_datasets.sh",
        "conformance_matrix.sh",
        "calibrate_d3.sh",
        "cpu_array.sh",
    ):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(
            include_gpu=False,
            prepare_containers=True,
            download_datasets=True,
            preprocess_datasets=True,
        ),
        seed=42,
        dry_run=True,
    )
    by_key = {step["key"]: step for step in summary["steps"]}

    assert by_key["prepare_containers"]["command"][-1] == str((slurm_dir / "prepare_containers.sh").resolve())
    assert by_key["download_datasets"]["command"][2:4] == ["--dependency", "afterok:<PREPARE_CONTAINERS_JOB_ID>"]
    assert by_key["preprocess_datasets"]["command"][2:4] == [
        "--dependency",
        "afterok:<DOWNLOAD_DATASETS_JOB_ID>:<PREPARE_CONTAINERS_JOB_ID>",
    ]
    assert by_key["conformance"]["command"][2:4] == [
        "--dependency",
        "afterok:<PREPROCESS_DATASETS_JOB_ID>:<PREPARE_CONTAINERS_JOB_ID>",
    ]
    assert by_key["calibrate"]["command"][2:4] == ["--dependency", "afterok:<CONFORMANCE_JOB_ID>"]


def test_submit_steps_exports_scenario_config_dir_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
        assert f"MAXIONBENCH_SLURM_DIR={slurm_dir.resolve()}" in export_value
        assert "MAXIONBENCH_SEED=42" in export_value
        assert "MAXIONBENCH_SCENARIO_CONFIG_DIR=configs/scenarios_paper" in export_value


def test_submit_steps_rejects_unknown_local_slurm_profile(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
    conformance:
      - ["--gres", "gpu:1"]
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

    cpu_cmd = [str(item) for item in summary["steps"][2]["command"]]
    assert "--partition" in cpu_cmd
    assert "--account" in cpu_cmd
    assert "--gres" not in cpu_cmd


def test_submit_steps_warns_on_unknown_step_override_keys(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
    caplog,
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh", "download_datasets.sh"):
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
  step_overrides:
    downlod_datasets:
      - ["--cpus-per-task", "4"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(overrides_path))

    caplog.set_level("WARNING")
    submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False, download_datasets=True),
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )

    assert "Ignoring unknown your_cluster.step_overrides keys: downlod_datasets" in caplog.text
    assert "Known step keys:" in caplog.text


def test_submit_steps_exports_output_root_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
        assert f"MAXIONBENCH_SLURM_DIR={slurm_dir.resolve()}" in export_value
        assert "MAXIONBENCH_OUTPUT_ROOT=artifacts/workstation_runs/example/results/slurm" in export_value


def test_submit_steps_exports_container_runtime_settings_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
        assert f"MAXIONBENCH_SLURM_DIR={slurm_dir.resolve()}" in export_value
        assert "MAXIONBENCH_CONTAINER_RUNTIME=apptainer" in export_value
        assert "MAXIONBENCH_CONTAINER_IMAGE=/shared/containers/maxionbench.sif" in export_value
        assert "MAXIONBENCH_CONTAINER_BIND=/shared/datasets|/shared/models/hf:/shared/models/hf" in export_value
        assert "MAXIONBENCH_HF_CACHE_DIR=/shared/models/hf" in export_value


def test_submit_steps_exports_conformance_matrix_path(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False),
        seed=42,
        conformance_matrix_path="artifacts/slurm_manifests/latest/conformance/conformance_matrix.csv",
        dry_run=True,
    )
    for step in summary["steps"]:
        cmd = [str(item) for item in step["command"]]
        export_value = cmd[cmd.index("--export") + 1]
        assert "MAXIONBENCH_CONFORMANCE_MATRIX=artifacts/slurm_manifests/latest/conformance/conformance_matrix.csv" in export_value


def test_validate_full_matrix_contract_rejects_gpu_omission() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/generated",
        cpu_rows=[],
        gpu_rows=[],
        selected_engines=[],
        selected_templates=[],
    )
    try:
        validate_full_matrix_contract(
            manifest=manifest,
            skip_gpu=True,
            prefetch_datasets=True,
            allow_gpu_unavailable_env="0",
        )
    except ValueError as exc:
        assert "--skip-gpu" in str(exc)
    else:
        raise AssertionError("expected validate_full_matrix_contract to reject GPU omission")


def test_validate_full_matrix_contract_rejects_reduced_gpu_row_count() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/generated",
        cpu_rows=[],
        gpu_rows=[
            RunManifestRow(
                group="gpu",
                config_path=f"/repo/generated/gpu_{idx}.yaml",
                engine="faiss-gpu",
                scenario="s5_rerank",
                dataset_bundle="D4",
                template_name=f"gpu_{idx}.yaml",
            )
            for idx in range(18)
        ],
        selected_engines=[],
        selected_templates=[],
    )
    try:
        validate_full_matrix_contract(
            manifest=manifest,
            skip_gpu=False,
            prefetch_datasets=True,
            allow_gpu_unavailable_env="0",
        )
    except ValueError as exc:
        assert "19 GPU rows" in str(exc)
    else:
        raise AssertionError("expected validate_full_matrix_contract to reject a reduced GPU manifest")


def test_validate_full_matrix_contract_allows_reduced_smoke_matrix() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/generated",
        cpu_rows=[],
        gpu_rows=[
            RunManifestRow(
                group="gpu",
                config_path="/repo/generated/gpu_0.yaml",
                engine="faiss-gpu",
                scenario="s5_rerank",
                dataset_bundle="D4",
                template_name="s5_rerank.yaml",
            )
        ],
        selected_engines=["faiss-gpu"],
        selected_templates=["s5_rerank.yaml"],
    )

    validate_full_matrix_contract(
        manifest=manifest,
        skip_gpu=False,
        prefetch_datasets=True,
        allow_gpu_unavailable_env="0",
        allow_reduced_matrix=True,
    )


def test_validate_full_matrix_contract_allows_reduced_cpu_only_smoke_matrix() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/generated",
        cpu_rows=[
            RunManifestRow(
                group="cpu",
                config_path="/repo/generated/cpu_0.yaml",
                engine="pgvector",
                scenario="s1_ann_frontier",
                dataset_bundle="D3",
                template_name="s1_ann_frontier_d3.yaml",
            )
        ],
        gpu_rows=[],
        selected_engines=["pgvector"],
        selected_templates=["s1_ann_frontier_d3.yaml"],
    )

    validate_full_matrix_contract(
        manifest=manifest,
        skip_gpu=False,
        prefetch_datasets=True,
        allow_gpu_unavailable_env="0",
        allow_reduced_matrix=True,
    )


def test_cli_submit_slurm_plan_forwards_allow_reduced_matrix(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_submit(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr("maxionbench.orchestration.slurm.submit_plan.main", _fake_submit)

    rc = cli_main(
        [
            "submit-slurm-plan",
            "--slurm-dir",
            "maxionbench/orchestration/slurm",
            "--scenario-config-dir",
            "configs/scenarios_paper",
            "--full-matrix",
            "--prefetch-datasets",
            "--allow-reduced-matrix",
        ]
    )

    assert rc == 0
    assert "--allow-reduced-matrix" in captured["argv"]


def test_submit_steps_requires_container_image_when_runtime_enabled(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
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
        prepare_containers=True,
        download_datasets=True,
        preprocess_datasets=True,
        include_postprocess=True,
        manifest=manifest,
    )
    by_key = {step.key: step for step in steps}

    assert [step.key for step in steps] == [
        "prepare_containers",
        "download_datasets",
        "preprocess_datasets",
        "conformance",
        "calibrate",
        "cpu_d3_baseline",
        "cpu_d3_workloads",
        "cpu_non_d3",
        "gpu_non_d3",
        "postprocess",
    ]
    assert by_key["prepare_containers"].depends_on == ()
    assert by_key["download_datasets"].depends_on == ("prepare_containers",)
    assert by_key["preprocess_datasets"].depends_on == ("download_datasets", "prepare_containers")
    assert by_key["conformance"].depends_on == ("preprocess_datasets", "prepare_containers")
    assert by_key["calibrate"].depends_on == ("conformance",)
    assert by_key["cpu_d3_baseline"].array == "0"
    assert by_key["cpu_d3_workloads"].array == "1"
    assert by_key["cpu_non_d3"].array == "2"
    assert by_key["gpu_non_d3"].array == "0"
    assert by_key["postprocess"].depends_on == (
        "cpu_d3_baseline",
        "cpu_d3_workloads",
        "cpu_non_d3",
        "gpu_non_d3",
    )


def test_build_submit_steps_splits_gpu_manifest_d3_dependency_chain() -> None:
    manifest = RunManifest(
        repo_root="/repo",
        generated_config_dir="/repo/artifacts/slurm_manifests/latest/generated_configs",
        cpu_rows=[],
        gpu_rows=[
            RunManifestRow(
                group="gpu",
                config_path="/repo/generated/s1_d3_faiss_gpu.yaml",
                engine="faiss-gpu",
                scenario="s1_ann_frontier",
                dataset_bundle="D3",
                template_name="s1_ann_frontier_d3.yaml",
            ),
            RunManifestRow(
                group="gpu",
                config_path="/repo/generated/s3_faiss_gpu.yaml",
                engine="faiss-gpu",
                scenario="s3_churn_smooth",
                dataset_bundle="D3",
                template_name="s3_churn_smooth.yaml",
            ),
            RunManifestRow(
                group="gpu",
                config_path="/repo/generated/s5_qdrant.yaml",
                engine="qdrant",
                scenario="s5_rerank",
                dataset_bundle="D4",
                template_name="s5_rerank.yaml",
            ),
        ],
        selected_engines=["faiss-gpu", "qdrant"],
        selected_templates=["s1_ann_frontier_d3.yaml", "s3_churn_smooth.yaml", "s5_rerank.yaml"],
    )

    steps = build_submit_steps(include_gpu=True, manifest=manifest, include_postprocess=True)
    by_key = {step.key: step for step in steps}

    assert [step.key for step in steps] == [
        "conformance",
        "calibrate",
        "gpu_d3_baseline",
        "gpu_d3_workloads",
        "gpu_non_d3",
        "postprocess",
    ]
    assert by_key["conformance"].depends_on == ()
    assert by_key["gpu_d3_baseline"].array == "0"
    assert by_key["gpu_d3_baseline"].depends_on == ("calibrate",)
    assert by_key["gpu_d3_workloads"].array == "1"
    assert by_key["gpu_d3_workloads"].depends_on == ("calibrate", "gpu_d3_baseline")
    assert by_key["gpu_non_d3"].array == "2"
    assert by_key["gpu_non_d3"].depends_on == ("calibrate",)
    assert by_key["postprocess"].depends_on == (
        "gpu_d3_baseline",
        "gpu_d3_workloads",
        "gpu_non_d3",
    )


def test_submit_steps_applies_gpu_all_profile_override_to_split_gpu_steps(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("calibrate_d3.sh", "gpu_array.sh"):
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
        steps=[
            SubmitStep(key="calibrate", script_name="calibrate_d3.sh"),
            SubmitStep(key="gpu_d3_baseline", script_name="gpu_array.sh"),
            SubmitStep(key="gpu_d3_workloads", script_name="gpu_array.sh", depends_on=("calibrate",)),
            SubmitStep(key="gpu_non_d3", script_name="gpu_array.sh", depends_on=("calibrate",)),
        ],
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )
    by_key = {step["key"]: [str(item) for item in step["command"]] for step in summary["steps"]}

    assert by_key["calibrate"][by_key["calibrate"].index("--gres") + 1] == "gpu:1"
    assert by_key["gpu_d3_baseline"][by_key["gpu_d3_baseline"].index("--gres") + 1] == "gpu:1"
    assert by_key["gpu_d3_workloads"][by_key["gpu_d3_workloads"].index("--gres") + 1] == "gpu:1"
    assert by_key["gpu_non_d3"][by_key["gpu_non_d3"].index("--gres") + 1] == "gpu:1"


def test_submit_steps_step_overrides_replace_duplicate_base_flags(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "conformance_matrix.sh",
        "calibrate_d3.sh",
        "cpu_array.sh",
        "gpu_array.sh",
        "postprocess.sh",
    ):
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
  step_overrides:
    postprocess:
      - ["--cpus-per-task", "8"]
      - ["--mem", "64G"]
      - ["--time", "01:00:00"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(overrides_path))

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=False, include_postprocess=True),
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )
    by_key = {step["key"]: [str(item) for item in step["command"]] for step in summary["steps"]}
    postprocess_cmd = by_key["postprocess"]

    assert postprocess_cmd.count("--cpus-per-task") == 1
    assert postprocess_cmd[postprocess_cmd.index("--cpus-per-task") + 1] == "8"
    assert postprocess_cmd.count("--mem") == 1
    assert postprocess_cmd[postprocess_cmd.index("--mem") + 1] == "64G"
    assert postprocess_cmd.count("--time") == 1
    assert postprocess_cmd[postprocess_cmd.index("--time") + 1] == "01:00:00"
    assert postprocess_cmd[postprocess_cmd.index("--account") + 1] == "private-account"


def test_submit_steps_dry_run_emits_no_duplicate_sbatch_flags_per_step(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "download_datasets.sh",
        "preprocess_datasets.sh",
        "conformance_matrix.sh",
        "calibrate_d3.sh",
        "cpu_array.sh",
        "gpu_array.sh",
        "postprocess.sh",
    ):
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
  step_overrides:
    download_datasets:
      - ["--cpus-per-task", "4"]
      - ["--mem", "32G"]
      - ["--time", "04:00:00"]
    preprocess_datasets:
      - ["--cpus-per-task", "32"]
      - ["--mem", "128G"]
      - ["--time", "08:00:00"]
    conformance:
      - ["--gres", "gpu:1"]
    calibrate:
      - ["--gres", "gpu:1"]
    gpu_all:
      - ["--gres", "gpu:1"]
    postprocess:
      - ["--cpus-per-task", "8"]
      - ["--mem", "64G"]
      - ["--time", "01:00:00"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(overrides_path))

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(
            include_gpu=True,
            download_datasets=True,
            preprocess_datasets=True,
            include_postprocess=True,
        ),
        seed=42,
        slurm_profile="your_cluster",
        dry_run=True,
    )

    for step in summary["steps"]:
        command = [str(item) for item in step["command"]]
        flags = [item for item in command if item.startswith("--")]
        duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
        assert duplicates == [], f"{step['key']} duplicated flags: {duplicates} in {command}"


def test_submit_steps_exports_run_manifest_when_provided(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "download_datasets.sh",
        "preprocess_datasets.sh",
        "conformance_matrix.sh",
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
        assert f"MAXIONBENCH_SLURM_DIR={slurm_dir.resolve()}" in export_value
        assert "MAXIONBENCH_SLURM_RUN_MANIFEST=artifacts/slurm_manifests/latest/run_manifest.json" in export_value


def test_submit_steps_uses_tracked_cluster_profile_examples(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(tmp_path / "missing_profiles_local.yaml"))
    monkeypatch.setenv("MAXIONBENCH_SLURM_ACCOUNT", "tracked-account")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        slurm_profile="cluster_a_apptainer",
        dry_run=True,
    )
    conformance_cmd = [str(item) for item in summary["steps"][0]["command"]]
    assert "--job-name" in conformance_cmd
    assert conformance_cmd[conformance_cmd.index("--job-name") + 1] == "maxion"
    assert "--cpus-per-task" in conformance_cmd
    assert conformance_cmd[conformance_cmd.index("--cpus-per-task") + 1] == "96"
    assert "--gres" in conformance_cmd
    assert conformance_cmd[conformance_cmd.index("--gres") + 1] == "gpu:1"


def test_submit_steps_expands_env_placeholders_in_tracked_profiles(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    for script_name in ("conformance_matrix.sh", "calibrate_d3.sh", "cpu_array.sh", "gpu_array.sh"):
        (slurm_dir / script_name).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.setenv("MAXIONBENCH_SLURM_PROFILE_OVERRIDES", str(tmp_path / "missing_profiles_local.yaml"))
    monkeypatch.setenv("MAXIONBENCH_SLURM_ACCOUNT", "tracked-account")
    monkeypatch.setenv("MAXIONBENCH_SLURM_PARTITION", "tracked-partition")

    summary = submit_steps(
        slurm_dir=slurm_dir,
        steps=build_submit_steps(include_gpu=True),
        seed=42,
        slurm_profile="cluster_b_apptainer",
        dry_run=True,
    )
    conformance_cmd = [str(item) for item in summary["steps"][0]["command"]]
    assert "--account" in conformance_cmd
    assert conformance_cmd[conformance_cmd.index("--account") + 1] == "tracked-account"
    assert "--partition" in conformance_cmd
    assert conformance_cmd[conformance_cmd.index("--partition") + 1] == "tracked-partition"
