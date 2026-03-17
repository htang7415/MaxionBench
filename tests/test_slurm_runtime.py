from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import yaml

from maxionbench.orchestration.slurm import preflight as preflight_mod
from maxionbench.orchestration.slurm.preflight import evaluate_preflight
from maxionbench.runtime.ports import allocate_named_ports, allocate_port, allocate_port_range


def test_allocate_port_range_and_named_ports() -> None:
    ports = allocate_port_range(count=3, base=25000, span=1000, offset=10)
    assert len(ports) == 3
    assert ports[1] == ports[0] + 1

    named = allocate_named_ports(["a", "b", "a"], base=26000, span=1000)
    assert set(named.keys()) == {"a", "b"}
    assert named["b"] == named["a"] + 1


def test_allocate_port_uses_array_task_id(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    first = allocate_port(base=27000, span=1000)

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    second = allocate_port(base=27000, span=1000)

    assert first != second
    assert second == first + 1


def test_preflight_uses_manifest_when_available(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "D3"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["fallback_config"] == "configs/scenarios/s2_filtered_ann.yaml"


def test_preflight_estimate_without_manifest(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_unknown.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "UNKNOWN"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["engine_bytes"] > 0
    assert summary["temp_bytes"] > 0


def test_preflight_verifies_dataset_cache_checksum_when_provided(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample.bin"
    dataset_file.write_bytes(b"cache-check")
    checksum = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": checksum,
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_ok.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is True
    assert int(summary["integrity_error_count"]) == 0
    assert int(summary["integrity_checked_files"]) == 1


def test_preflight_fails_when_dataset_cache_checksum_mismatches(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample_bad.bin"
    dataset_file.write_bytes(b"cache-check-bad")
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": ("0" * 64),
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_bad.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is False
    assert int(summary["integrity_error_count"]) >= 1
    assert summary["ok"] is False
    assert any("sha256 mismatch" in msg for msg in summary["integrity_errors"])


def test_preflight_fails_when_known_bundle_manifest_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_missing_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(preflight_mod, "_load_manifest", lambda _: None)
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "missing manifest" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_preflight_fails_when_known_bundle_manifest_has_invalid_size_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_bad_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(
        preflight_mod,
        "_load_manifest",
        lambda _: {
            "dataset_bundle": "D3",
            "approx_bytes_dataset": 0,
            "approx_bytes_engine": 100,
            "approx_bytes_temp": 10,
        },
    )
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "must be > 0" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_slurm_common_runs_pre_run_gate_before_runner() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_PRE_RUN_GATE" in text
    assert "MAXIONBENCH_ALLOW_GPU_UNAVAILABLE" in text
    assert "MAXIONBENCH_CONFORMANCE_MATRIX" in text
    assert "MAXIONBENCH_OUTPUT_ROOT" in text
    assert "MAXIONBENCH_DATASET_ROOT" in text
    assert "MAXIONBENCH_DATASET_CACHE_DIR" in text
    assert "MAXIONBENCH_FIGURES_ROOT" in text
    assert "MAXIONBENCH_D3_PARAMS_PATH" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "MAXIONBENCH_CONFORMANCE_TIMEOUT_S" in text
    assert "MAXIONBENCH_SERVICE_START_GRACE_S" in text
    assert "MAXIONBENCH_SERVICE_START_POLL_S" in text
    assert "MAXIONBENCH_SERVICE_LOG_TAIL_LINES" in text
    assert "MAXIONBENCH_CONTAINER_RUNTIME" in text
    assert "MAXIONBENCH_CONTAINER_IMAGE" in text
    assert "MAXIONBENCH_CONTAINER_BIND" in text
    assert "MAXIONBENCH_HF_CACHE_DIR" in text
    assert "MAXIONBENCH_APPTAINER_MODULE" in text
    assert "MAXIONBENCH_MODULE_INIT_SH" in text
    assert "MAXIONBENCH_CLEANUP_LOCAL_SCRATCH" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "MAXIONBENCH_QDRANT_IMAGE" in text
    assert "MAXIONBENCH_PGVECTOR_IMAGE" in text
    assert "MAXIONBENCH_OPENSEARCH_IMAGE" in text
    assert "MAXIONBENCH_WEAVIATE_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_ETCD_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_MINIO_IMAGE" in text
    assert "MAXIONBENCH_MILVUS_IMAGE" in text
    assert "mb_source_dataset_env()" in text
    assert "mb_require_dataset_env_contract()" in text
    assert "mb_require_gpu_fail_fast()" in text
    assert "mb_require_visible_gpu()" in text
    assert "mb_ensure_apptainer()" in text
    assert "module load" in text
    assert "apptainer exec --cleanenv" in text
    assert "PYTHONNOUSERSITE=1" in text
    assert "python -s" in text
    assert "mb_python()" in text
    assert "mb_cleanup_local_runtime()" in text
    assert "expand_env_placeholders" in text
    assert "mb_read_config_field()" in text
    gate_marker = "pre-run-gate"
    runner_marker = "python -m maxionbench.orchestration.runner"
    assert gate_marker in text
    assert runner_marker in text
    assert text.index(gate_marker) < text.index(runner_marker)


def test_slurm_common_revalidates_fallback_config_after_preflight_failure() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert 'mb_log "scratch preflight failed, validating fallback config ${resolved_fallback}"' in text
    assert 'if mb_run_scratch_preflight "${resolved_fallback}"; then' in text
    assert 'mb_log "fallback config ${resolved_fallback} also failed scratch preflight"' in text


def test_slurm_common_has_managed_engine_service_lifecycle_helpers() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "mb_detect_engine_runtime_mode()" in text
    assert "mb_engine_requires_service()" in text
    assert "mb_start_engine_services()" in text
    assert "mb_stop_engine_services()" in text
    assert "mb_wait_engine_health()" in text
    assert "mb_start_qdrant_service()" in text
    assert "mb_start_pgvector_service()" in text
    assert "mb_start_opensearch_service()" in text
    assert "mb_start_weaviate_service()" in text
    assert "mb_start_milvus_services()" in text
    assert "mb_start_apptainer_service_process()" in text
    assert "mb_validate_apptainer_service_image()" in text
    assert "mb_wait_named_adapter_health()" in text
    assert "mb_capture_local_diagnostics()" in text
    assert "mb_finalize_job()" in text
    assert 'MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${SLURM_TMPDIR}/lancedb/service"' in text
    assert "MAXIONBENCH_PGVECTOR_DSN=" in text
    assert "apptainer inspect" in text
    assert "command -v" in text
    assert "mb_log_file_tail" in text


def test_slurm_common_loads_apptainer_module_when_binary_missing(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_log = tmp_path / "apptainer.log"
    fake_image = tmp_path / "maxionbench.sif"
    fake_image.write_text("image\n", encoding="utf-8")

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_TEST_APPTAINER_LOG}"
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    module_init = tmp_path / "modules.sh"
    module_init.write_text(
        f"""module() {{
  if [[ "${{1:-}}" == "load" && "${{2:-}}" == "apptainer" ]]; then
    export PATH="{fake_bin_dir}:$PATH"
    return 0
  fi
  return 1
}}
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export PATH="/usr/bin:/bin"; '
                f'export MAXIONBENCH_CONTAINER_IMAGE="{fake_image}"; '
                'export MAXIONBENCH_CONTAINER_RUNTIME="apptainer"; '
                'export MAXIONBENCH_APPTAINER_MODULE="apptainer"; '
                f'export MAXIONBENCH_MODULE_INIT_SH="{module_init}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{fake_log}"; '
                'mb_python -V'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "apptainer not found in PATH; attempting module bootstrap with apptainer" in completed.stderr
    assert f"sourced module init {module_init}" in completed.stderr
    assert "loading apptainer module apptainer" in completed.stderr
    assert f"using apptainer binary {fake_apptainer}" in completed.stderr
    logged_args = fake_log.read_text(encoding="utf-8")
    assert "exec" in logged_args
    assert "--cleanenv" in logged_args
    assert str(fake_image) in logged_args
    assert "PYTHONNOUSERSITE=1" in logged_args
    assert "python -s -V" in logged_args


def test_stage_config_command_substitution_stays_clean_with_apptainer(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_log = tmp_path / "apptainer_stage.log"
    fake_image = tmp_path / "maxionbench.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    slurm_tmpdir = tmp_path / "slurm_tmp"
    dataset_source = tmp_path / "real_d3.npy"
    dataset_source.write_bytes(b"vectors\n")
    config_path = tmp_path / "stage_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario": "cleanup_probe",
                "dataset_bundle": "D3",
                "dataset_path": "${MAXIONBENCH_D3_DATASET_PATH}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_TEST_APPTAINER_LOG}"
if [[ "${1:-}" == "inspect" ]]; then
  exit 0
fi
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -ge ${#args[@]} ]]; then
  exit 0
fi
index=$((index + 1))
exec "${args[@]:${index}}"
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_CONTAINER_IMAGE="{fake_image}"; '
                'export MAXIONBENCH_CONTAINER_RUNTIME="apptainer"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{fake_log}"; '
                f'export MAXIONBENCH_D3_DATASET_PATH="{dataset_source}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'STAGED_CONFIG="$(mb_stage_config_to_tmp "{config_path}")"; '
                'printf "STAGED=%s\\n" "${STAGED_CONFIG}"; '
                'test -f "${STAGED_CONFIG}"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    staged_path = Path(stdout_lines["STAGED"])
    assert staged_path == slurm_tmpdir / "maxionbench_stage" / "4242_7" / "config.yaml"
    assert staged_path.exists()
    assert "[maxionbench]" not in stdout_lines["STAGED"]
    staged_payload = yaml.safe_load(staged_path.read_text(encoding="utf-8"))
    assert staged_payload["dataset_path"] == str(
        slurm_tmpdir / "maxionbench_stage" / "4242_7" / "datasets" / "dataset" / dataset_source.name
    )
    assert Path(staged_payload["dataset_path"]).exists()
    assert f"using apptainer binary {fake_apptainer}" in completed.stderr


def test_slurm_common_passes_service_env_inside_apptainer_exec(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_image = tmp_path / "qdrant.sif"
    fake_image.write_text("image\n", encoding="utf-8")
    full_log = tmp_path / "apptainer_service.log"
    post_image_log = tmp_path / "apptainer_service_post_image.log"
    slurm_tmpdir = tmp_path / "slurm_tmp"

    fake_apptainer = fake_bin_dir / "apptainer"
    fake_apptainer.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "${MAXIONBENCH_TEST_APPTAINER_LOG}"
args=("$@")
index=0
while [[ ${index} -lt ${#args[@]} ]]; do
  case "${args[${index}]}" in
    exec|--cleanenv|--nv)
      index=$((index + 1))
      ;;
    --bind)
      index=$((index + 2))
      ;;
    *)
      break
      ;;
  esac
done
if [[ ${index} -lt ${#args[@]} ]]; then
  index=$((index + 1))
fi
printf '%s\\n' "${args[@]:${index}}" > "${MAXIONBENCH_TEST_POST_IMAGE_LOG}"
if [[ "${args[${index}]:-}" == "/bin/sh" && "${args[$((index + 1))]:-}" == "-lc" && "${args[$((index + 2))]:-}" == command\ -v* ]]; then
  exit 0
fi
sleep 30
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                f'export PATH="{fake_bin_dir}:$PATH"; '
                f'export MAXIONBENCH_QDRANT_IMAGE="{fake_image}"; '
                f'export MAXIONBENCH_TEST_APPTAINER_LOG="{full_log}"; '
                f'export MAXIONBENCH_TEST_POST_IMAGE_LOG="{post_image_log}"; '
                f'export SLURM_TMPDIR="{slurm_tmpdir}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                'mb_require_tmpdir; '
                'mb_allocate_ports; '
                'mb_start_qdrant_service; '
                'mb_stop_engine_services'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    post_image_args = post_image_log.read_text(encoding="utf-8")
    assert post_image_args.startswith("env\n")
    assert "QDRANT__SERVICE__HOST=0.0.0.0" in post_image_args
    assert "QDRANT__SERVICE__HTTP_PORT=" in post_image_args
    assert "QDRANT__SERVICE__GRPC_PORT=" in post_image_args
    assert "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage" in post_image_args
    assert "/bin/sh" in post_image_args


def test_slurm_common_cleanup_local_runtime_removes_scratch_but_keeps_final_output(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    config_path = tmp_path / "cleanup_config.yaml"
    dataset_dir = tmp_path / "processed_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "base.npy").write_bytes(b"123")
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario": "cleanup_probe",
                "processed_dataset_path": str(dataset_dir),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scratch_dir = tmp_path / "scratch"
    output_root = tmp_path / "results"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'export SLURM_TMPDIR="{scratch_dir}"; '
                f'export MAXIONBENCH_OUTPUT_ROOT="{output_root}"; '
                'export MAXIONBENCH_CLEANUP_LOCAL_SCRATCH="1"; '
                'export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="${SLURM_TMPDIR}/lancedb/service"; '
                'mb_require_tmpdir; '
                'mb_prepare_output_paths "cleanup_probe"; '
                f'STAGED_CONFIG="$(mb_stage_config_to_tmp "{config_path}")"; '
                'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"; '
                'mkdir -p "$(mb_engine_runtime_root)/logs" "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI}" "${MB_OUTPUT_TMP}"; '
                'printf "service\\n" > "$(mb_engine_runtime_root)/logs/service.log"; '
                'printf "result\\n" > "${MB_OUTPUT_TMP}/results.parquet"; '
                'mb_copy_back_output; '
                'mb_cleanup_local_runtime; '
                'printf "FINAL=%s\\n" "${MB_OUTPUT_FINAL}"; '
                'printf "TMP_EXISTS=%s\\n" "$(test -e "${MB_OUTPUT_TMP}" && echo 1 || echo 0)"; '
                'printf "STAGE_EXISTS=%s\\n" "$(test -e "${MB_STAGE_ROOT}" && echo 1 || echo 0)"; '
                'printf "RUNTIME_EXISTS=%s\\n" "$(test -e "$(mb_engine_runtime_root)" && echo 1 || echo 0)"; '
                'printf "LANCEDB_EXISTS=%s\\n" "$(test -e "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI}" && echo 1 || echo 0)"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    final_output = Path(stdout_lines["FINAL"])
    assert final_output.exists()
    assert (final_output / "results.parquet").read_text(encoding="utf-8") == "result\n"
    assert stdout_lines["TMP_EXISTS"] == "0"
    assert stdout_lines["STAGE_EXISTS"] == "0"
    assert stdout_lines["RUNTIME_EXISTS"] == "0"
    assert stdout_lines["LANCEDB_EXISTS"] == "0"


def test_slurm_common_finalize_job_captures_runtime_logs_before_cleanup(tmp_path: Path) -> None:
    common_path = Path("maxionbench/orchestration/slurm/common.sh").resolve()
    scratch_dir = tmp_path / "scratch"
    output_root = tmp_path / "results"

    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f'source "{common_path}"; '
                'export SLURM_JOB_ID="4242"; '
                'export SLURM_ARRAY_TASK_ID="7"; '
                f'export SLURM_TMPDIR="{scratch_dir}"; '
                f'export MAXIONBENCH_OUTPUT_ROOT="{output_root}"; '
                'export MAXIONBENCH_CLEANUP_LOCAL_SCRATCH="1"; '
                'mb_require_tmpdir; '
                'mb_prepare_output_paths "finalize_probe"; '
                'mkdir -p "$(mb_engine_runtime_root)/logs" "${MB_OUTPUT_TMP}"; '
                'printf "service\\n" > "$(mb_engine_runtime_root)/logs/service.log"; '
                'printf "result\\n" > "${MB_OUTPUT_TMP}/results.parquet"; '
                'mb_finalize_job 9 0; '
                'printf "FINAL=%s\\n" "${MB_OUTPUT_FINAL}"; '
                'printf "RUNTIME_EXISTS=%s\\n" "$(test -e "$(mb_engine_runtime_root)" && echo 1 || echo 0)"'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stdout_lines = dict(
        line.split("=", maxsplit=1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    final_output = Path(stdout_lines["FINAL"])
    assert final_output.exists()
    assert (final_output / "results.parquet").read_text(encoding="utf-8") == "result\n"
    service_logs = list(final_output.glob("logs/local_runtime/engine_runtime/**/service.log"))
    assert service_logs, list(final_output.rglob("*"))
    assert service_logs[0].read_text(encoding="utf-8") == "service\n"
    assert stdout_lines["RUNTIME_EXISTS"] == "0"


def test_slurm_wrapper_scripts_source_common_from_exported_slurm_dir() -> None:
    for rel_path in (
        "maxionbench/orchestration/slurm/download_datasets.sh",
        "maxionbench/orchestration/slurm/preprocess_datasets.sh",
        "maxionbench/orchestration/slurm/conformance_matrix.sh",
        "maxionbench/orchestration/slurm/postprocess.sh",
        "maxionbench/orchestration/slurm/calibrate_d3.sh",
        "maxionbench/orchestration/slurm/prefetch_datasets.sh",
        "maxionbench/orchestration/slurm/cpu_array.sh",
        "maxionbench/orchestration/slurm/gpu_array.sh",
    ):
        text = Path(rel_path).read_text(encoding="utf-8")
        assert 'SLURM_DIR="${MAXIONBENCH_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"' in text
        assert 'source "${SLURM_DIR}/common.sh"' in text


def test_cpu_array_includes_d3_matched_s1_baseline_config() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "configs/scenarios/s1_ann_frontier_d3.yaml" in text


def test_cpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "run_manifest resolve" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text
    assert 'SCENARIO_KEY="$(mb_read_config_field "${CONFIG_PATH}" "scenario")"' in text


def test_cpu_array_supports_skip_s6_env_flag() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_S6" in text
    assert "s6_fusion.yaml" in text
    assert "skipping S6 task index" in text


def test_cpu_array_starts_and_stops_managed_engine_services() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert 'if mb_engine_requires_service "${STAGED_CONFIG}"; then' in text
    assert "mb_start_engine_services" in text
    assert "mb_wait_engine_health" in text
    assert 'trap \'status=$?; trap - EXIT; mb_finalize_job "${status}" "${SERVICE_STARTED:-0}"; exit "${status}"\' EXIT' in text
    assert "SERVICE_STARTED=1" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_gpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert "MAXIONBENCH_SLURM_RUN_MANIFEST" in text
    assert "run_manifest resolve" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text


def test_gpu_array_starts_and_stops_managed_engine_services() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert 'if mb_engine_requires_service "${STAGED_CONFIG}"; then' in text
    assert "mb_start_engine_services" in text
    assert "mb_wait_engine_health" in text
    assert 'trap \'status=$?; trap - EXIT; mb_finalize_job "${status}" "${SERVICE_STARTED:-0}"; exit "${status}"\' EXIT' in text
    assert "SERVICE_STARTED=1" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_new_slurm_pipeline_scripts_exist() -> None:
    for path in (
        Path("maxionbench/orchestration/slurm/download_datasets.sh"),
        Path("maxionbench/orchestration/slurm/preprocess_datasets.sh"),
        Path("maxionbench/orchestration/slurm/conformance_matrix.sh"),
        Path("maxionbench/orchestration/slurm/postprocess.sh"),
        Path("run_slurm_pipeline.sh"),
        Path("maxionbench/orchestration/slurm/profiles_clusters.example.yaml"),
        Path(".env.slurm.example"),
    ):
        assert path.exists(), path


def test_gpu_array_explicitly_lists_track_b_and_track_c_entries() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "s1_ann_frontier_track_b_gpu.yaml" in text
    assert "s5_rerank_track_c_gpu.yaml" in text


def test_calibrate_d3_supports_scenario_dir_override_with_explicit_override_precedence() -> None:
    text = Path("maxionbench/orchestration/slurm/calibrate_d3.sh").read_text(encoding="utf-8")
    assert 'CONFIG_PATH="${MAXIONBENCH_CALIBRATE_CONFIG:-configs/scenarios/calibrate_d3.yaml}"' in text
    assert 'if [[ -z "${MAXIONBENCH_CALIBRATE_CONFIG:-}" ]]; then' in text
    assert 'SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"' in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"' in text
    assert 'if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then' in text
    assert "mb_source_dataset_env" in text
    assert 'export MB_STAGE_ROOT="$(dirname "${STAGED_CONFIG}")"' in text
    assert "mb_finalize_job" in text


def test_prefetch_datasets_script_exists_and_uses_prefetch_helper() -> None:
    text = Path("maxionbench/orchestration/slurm/prefetch_datasets.sh").read_text(encoding="utf-8")
    assert "dataset_prefetch" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "mb_source_dataset_env" in text
