from __future__ import annotations

from pathlib import Path


def test_env_docker_example_matches_reference_service_defaults() -> None:
    env_text = Path(".env.docker.example").read_text(encoding="utf-8")
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "MAXIONBENCH_MILVUS_ETCD_IMAGE=quay.io/coreos/etcd:v3.5.18" in env_text
    assert "MAXIONBENCH_MILVUS_MINIO_IMAGE=minio/minio:RELEASE.2024-11-07T00-52-20Z" in env_text
    assert "MAXIONBENCH_MILVUS_IMAGE=milvusdb/milvus:v2.5.27" in env_text

    assert "${MAXIONBENCH_MILVUS_ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.18}" in compose_text
    assert "${MAXIONBENCH_MILVUS_MINIO_IMAGE:-minio/minio:RELEASE.2024-11-07T00-52-20Z}" in compose_text
    assert "${MAXIONBENCH_MILVUS_IMAGE:-milvusdb/milvus:v2.5.27}" in compose_text
