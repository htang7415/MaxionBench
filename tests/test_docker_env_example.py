from __future__ import annotations

from pathlib import Path


def test_env_docker_example_matches_portable_service_defaults() -> None:
    env_text = Path(".env.docker.example").read_text(encoding="utf-8")
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "MAXIONBENCH_QDRANT_IMAGE=qdrant/qdrant:v1.17.1" in env_text
    assert "MAXIONBENCH_PGVECTOR_IMAGE=pgvector/pgvector:0.8.2-pg16-trixie" in env_text

    assert "${MAXIONBENCH_QDRANT_IMAGE:-qdrant/qdrant:v1.17.1}" in compose_text
    assert "${MAXIONBENCH_PGVECTOR_IMAGE:-pgvector/pgvector:0.8.2-pg16-trixie}" in compose_text
    assert '"5432:5432"' in compose_text
