from __future__ import annotations

from pathlib import Path


def test_command_md_is_portable_agentic_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Portable-Agentic Commands" in text
    assert "`install -> conformance -> download -> one-time preprocess -> one-time embedding precompute -> docker compose up -> run-matrix -> execute-run-matrix -> report -> archive`" in text
    assert 'pip install -e ".[dev,engines,reporting,datasets,embeddings]"' in text
    assert "maxionbench conformance-matrix --config-dir configs/conformance --out-dir artifacts/conformance --timeout-s 30" in text
    assert "--datasets scifact,fiqa,crag,frames" in text
    assert "maxionbench preprocess-frames-portable" in text
    assert "maxionbench precompute-text-embeddings --input dataset/processed/D4 --model-id BAAI/bge-small-en-v1.5" in text
    assert "maxionbench precompute-text-embeddings --input dataset/processed/frames_portable --model-id BAAI/bge-base-en-v1.5" in text
    assert "docker compose up -d qdrant pgvector opensearch weaviate milvus" in text
    assert "maxionbench run-matrix \\" in text
    assert "--out-dir artifacts/run_matrix/portable_b0" in text
    assert "--output-root artifacts/runs/portable/b0" in text
    assert "maxionbench execute-run-matrix \\" in text
    assert "--matrix artifacts/run_matrix/portable_b0/run_matrix.json" in text
    assert "--budget b0" in text
    assert "--budget b1" in text
    assert "--budget b2" in text
    assert "configs/scenarios_portable/s1_single_hop.yaml" in text
    assert "configs/scenarios_portable/s2_streaming_memory.yaml" in text
    assert "configs/scenarios_portable/s3_multi_hop.yaml" in text
    assert 'tar -czf "results/${RUN_ID}.tar.gz" -C results "${RUN_ID}"' in text

    assert "run_docker_scenario.sh" not in text
    assert "run_workstation.sh" not in text
    assert "save_results_bundle.sh" not in text
