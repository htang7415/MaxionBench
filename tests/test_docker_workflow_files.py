from __future__ import annotations

from pathlib import Path
import subprocess


def test_docker_workflow_files_exist_and_reference_cpu_and_gpu_benchmarks() -> None:
    compose_path = Path("docker-compose.yml")
    run_script = Path("run_docker_scenario.sh")
    env_example = Path(".env.docker.example")
    gpu_dockerfile = Path("Dockerfile.gpu")

    assert compose_path.exists()
    assert run_script.exists()
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

    script_text = run_script.read_text(encoding="utf-8")
    assert "--benchmark-service" in script_text
    assert 'BENCHMARK_SERVICE="benchmark-gpu"' in script_text
    assert 'docker compose build "${BENCHMARK_SERVICE}"' in script_text
    assert 'OPENSEARCH_DATA_DIR="${MAXIONBENCH_OPENSEARCH_DATA_DIR:-${ROOT_DIR}/artifacts/containers/opensearch_data}"' in script_text
    assert 'mkdir -p "${OPENSEARCH_DATA_DIR}"' in script_text
    assert 'chmod 0777 "${OPENSEARCH_DATA_DIR}"' in script_text
    assert 'docker compose run --rm "${BENCHMARK_SERVICE}" wait-adapter --config "${CONFIG_PATH_CONTAINER}"' in script_text
    assert 'docker compose run --rm "${BENCHMARK_SERVICE}" run --config "${CONFIG_PATH_CONTAINER}"' in script_text
    assert 'CONFIG_PATH_CONTAINER="/workspace/' in script_text
    assert "SERVICES=(qdrant)" in script_text
    assert "SERVICES=(milvus)" in script_text
    assert "SERVICES=(opensearch)" in script_text
    assert "SERVICES=(pgvector)" in script_text
    assert "SERVICES=(weaviate)" in script_text


def test_docker_run_script_is_bash_parseable() -> None:
    completed = subprocess.run(
        ["bash", "-n", "run_docker_scenario.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
