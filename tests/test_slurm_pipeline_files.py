from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_slurm_pipeline_files_exist_and_reference_full_matrix_flow() -> None:
    script = Path("run_slurm_pipeline.sh")
    download = Path("maxionbench/orchestration/slurm/download_datasets.sh")
    preprocess = Path("maxionbench/orchestration/slurm/preprocess_datasets.sh")
    postprocess = Path("maxionbench/orchestration/slurm/postprocess.sh")
    profiles = Path("maxionbench/orchestration/slurm/profiles_clusters.example.yaml")
    env_example = Path(".env.slurm.example")

    for path in (script, download, preprocess, postprocess, profiles, env_example):
        assert path.exists(), path

    text = script.read_text(encoding="utf-8")
    assert "maxionbench submit-slurm-plan" in text
    assert "--download-datasets" in text
    assert "--preprocess-datasets" in text
    assert "--include-postprocess" in text
    assert "--full-matrix" in text
    assert "--container-runtime apptainer" in text
    assert "euler_apptainer" in text
    assert "nrel_apptainer" in text
    assert "--shared-root <path>" in text
    assert 'SUBMIT_ROOT="$(pwd -P)"' in text
    assert 'printf \'%s\\n\' "${SUBMIT_ROOT}"' in text

    download_text = download.read_text(encoding="utf-8")
    assert "maxionbench.cli download-datasets" in download_text
    assert "--root" in download_text
    assert "--cache-dir" in download_text

    preprocess_text = preprocess.read_text(encoding="utf-8")
    assert "preprocess-datasets ann-hdf5" in preprocess_text
    assert "preprocess-datasets d3-yfcc" in preprocess_text
    assert "preprocess-datasets beir" in preprocess_text
    assert "preprocess-datasets crag" in preprocess_text

    postprocess_text = postprocess.read_text(encoding="utf-8")
    assert "maxionbench.cli validate" in postprocess_text
    assert "maxionbench.cli report" in postprocess_text

    profiles_text = profiles.read_text(encoding="utf-8")
    assert "cluster_a_apptainer:" in profiles_text
    assert "cluster_b_apptainer:" in profiles_text
    assert "download_datasets:" in profiles_text
    assert "preprocess_datasets:" in profiles_text
    assert "postprocess:" in profiles_text

    env_text = env_example.read_text(encoding="utf-8")
    assert "MAXIONBENCH_SLURM_ACCOUNT=" in env_text
    assert "MAXIONBENCH_CONTAINER_IMAGE=" in env_text
    assert "MAXIONBENCH_QDRANT_IMAGE=" in env_text
    assert "MAXIONBENCH_PGVECTOR_IMAGE=" in env_text
    assert "MAXIONBENCH_OPENSEARCH_IMAGE=" in env_text
    assert "MAXIONBENCH_WEAVIATE_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_ETCD_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_MINIO_IMAGE=" in env_text
    assert "MAXIONBENCH_MILVUS_IMAGE=" in env_text
    assert "MAXIONBENCH_DATASET_ROOT=" in env_text
    assert "MAXIONBENCH_OUTPUT_ROOT=" in env_text


def test_slurm_pipeline_shell_scripts_are_bash_parseable() -> None:
    for path in (
        "run_slurm_pipeline.sh",
        "maxionbench/orchestration/slurm/download_datasets.sh",
        "maxionbench/orchestration/slurm/preprocess_datasets.sh",
        "maxionbench/orchestration/slurm/postprocess.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", path],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{path}: {completed.stdout}{completed.stderr}"


def test_run_slurm_pipeline_derives_cluster_storage_defaults(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    source_script = Path("run_slurm_pipeline.sh")
    script_path = repo_dir / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)
    submit_root = tmp_path / "submit_root"
    submit_root.mkdir(parents=True, exist_ok=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CACHE=%s\\n' "${MAXIONBENCH_DATASET_CACHE_DIR:-}"
  printf 'OUTPUT=%s\\n' "${MAXIONBENCH_OUTPUT_ROOT:-}"
  printf 'FIGURES=%s\\n' "${MAXIONBENCH_FIGURES_ROOT:-}"
  printf 'HF=%s\\n' "${MAXIONBENCH_HF_CACHE_DIR:-}"
  printf 'ARGS=%s\\n' "$*"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"
    env.pop("MAXIONBENCH_SHARED_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_ROOT", None)
    env.pop("MAXIONBENCH_DATASET_CACHE_DIR", None)
    env.pop("MAXIONBENCH_OUTPUT_ROOT", None)
    env.pop("MAXIONBENCH_FIGURES_ROOT", None)
    env.pop("MAXIONBENCH_HF_CACHE_DIR", None)

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--cluster",
            "euler",
            "--container-image",
            "/shared/containers/maxionbench.sif",
        ],
        cwd=submit_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    log_text = stub_log.read_text(encoding="utf-8")
    assert f"DATASET={submit_root}/dataset" in log_text
    assert f"CACHE={submit_root}/.cache" in log_text
    assert f"OUTPUT={submit_root}/results" in log_text
    assert f"FIGURES={submit_root}/figures" in log_text
    assert f"HF={submit_root}/.cache/huggingface" in log_text
    assert "ARGS=submit-slurm-plan" in log_text


def test_run_slurm_pipeline_shared_root_override_derives_all_paths(tmp_path: Path) -> None:
    source_script = Path("run_slurm_pipeline.sh")
    script_path = tmp_path / "run_slurm_pipeline.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub_log = tmp_path / "maxionbench_env.log"
    stub_path = bin_dir / "maxionbench"
    stub_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'DATASET=%s\\n' "${MAXIONBENCH_DATASET_ROOT:-}"
  printf 'CACHE=%s\\n' "${MAXIONBENCH_DATASET_CACHE_DIR:-}"
  printf 'OUTPUT=%s\\n' "${MAXIONBENCH_OUTPUT_ROOT:-}"
  printf 'FIGURES=%s\\n' "${MAXIONBENCH_FIGURES_ROOT:-}"
  printf 'HF=%s\\n' "${MAXIONBENCH_HF_CACHE_DIR:-}"
} > "${MAXIONBENCH_STUB_LOG}"
printf '{"ok": true}\\n'
""",
        encoding="utf-8",
    )
    stub_path.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MAXIONBENCH_STUB_LOG"] = str(stub_log)
    env["USER"] = "tester"

    completed = subprocess.run(
        [
            "bash",
            "run_slurm_pipeline.sh",
            "--cluster",
            "nrel",
            "--container-image",
            "/shared/containers/maxionbench.sif",
            "--shared-root",
            "/projects/demo/maxionbench",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    log_text = stub_log.read_text(encoding="utf-8")
    assert "DATASET=/projects/demo/maxionbench/dataset" in log_text
    assert "CACHE=/projects/demo/maxionbench/.cache" in log_text
    assert "OUTPUT=/projects/demo/maxionbench/results" in log_text
    assert "FIGURES=/projects/demo/maxionbench/figures" in log_text
    assert "HF=/projects/demo/maxionbench/.cache/huggingface" in log_text
