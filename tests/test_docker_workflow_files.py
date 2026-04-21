from __future__ import annotations

from pathlib import Path


def test_docker_workflow_assets_exist_and_reference_services() -> None:
    compose_path = Path("docker-compose.yml")
    env_example = Path(".env.docker.example")
    gpu_dockerfile = Path("Dockerfile.gpu")

    assert compose_path.exists()
    assert env_example.exists()
    assert gpu_dockerfile.exists()

    compose_text = compose_path.read_text(encoding="utf-8")
    assert "benchmark:" in compose_text
    assert "benchmark-gpu:" in compose_text
    assert "dockerfile: Dockerfile.gpu" in compose_text
    assert "gpus: all" in compose_text
    assert "qdrant:" in compose_text
    assert "pgvector:" in compose_text
    assert "opensearch:" in compose_text
    assert "weaviate:" in compose_text
    assert "milvus:" in compose_text
    assert "opensearchproject/opensearch:2.11.1" in compose_text
    assert "MAXIONBENCH_OPENSEARCH_DATA_DIR" in compose_text
    assert "./artifacts/containers/opensearch_data" in compose_text
    assert "MAXIONBENCH_QDRANT_HOST: qdrant" in compose_text
    assert "MAXIONBENCH_PGVECTOR_DSN: postgresql://postgres:postgres@pgvector:5432/postgres" in compose_text
    assert 'expose:\n      - "5432"' in compose_text
    assert '"5432:5432"' not in compose_text
    assert "MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI" in compose_text

    env_text = env_example.read_text(encoding="utf-8")
    assert "MAXIONBENCH_OPENSEARCH_IMAGE=opensearchproject/opensearch:2.11.1" in env_text
    assert "MAXIONBENCH_OPENSEARCH_DATA_DIR=./artifacts/containers/opensearch_data" in env_text

    gpu_text = gpu_dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in gpu_text
    assert 'python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"' in gpu_text
    assert 'python -m pip install "${MAXIONBENCH_FAISS_GPU_PIP_SPEC}"' in gpu_text
    assert 'python -m pip install --upgrade --force-reinstall \\' in gpu_text
    assert '"h5py>=3.11"' in gpu_text
    assert '"numpy>=2,<3"' in gpu_text
    assert not Path("run_docker_scenario.sh").exists()
