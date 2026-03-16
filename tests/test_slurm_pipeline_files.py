from __future__ import annotations

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
