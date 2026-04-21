from __future__ import annotations

from pathlib import Path


def test_command_md_is_portable_agentic_operator_doc() -> None:
    text = Path("command.md").read_text(encoding="utf-8")

    assert "# MaxionBench Portable-Agentic Commands" in text
    assert "install -> services -> conformance -> data+embeddings -> budgets -> report -> archive" in text
    assert 'pip install -e ".[dev,engines,reporting,datasets,embeddings]"' in text
    assert "maxionbench services up --profile portable --wait --json" in text
    assert "maxionbench conformance-matrix \\" in text
    assert "--adapters mock,faiss-cpu,lancedb-inproc,lancedb-service,qdrant,pgvector" in text
    assert "--datasets scifact,fiqa,crag,frames" in text
    assert "maxionbench preprocess-frames-portable" in text
    assert "for ROOT in dataset/processed/D4 dataset/processed/frames_portable; do" in text
    assert "maxionbench precompute-text-embeddings --input \"$ROOT\" --model-id BAAI/bge-small-en-v1.5" in text
    assert "maxionbench precompute-text-embeddings --input \"$ROOT\" --model-id BAAI/bge-base-en-v1.5" in text
    assert "for BUDGET in b0 b1 b2; do" in text
    assert "`configs/engines_portable` is the saved project engine set" in text
    assert "`run-matrix --budget` writes `budget_level` into generated configs" in text
    assert "maxionbench run-matrix \\" in text
    assert "--engine-config-dir configs/engines_portable" in text
    assert '--out-dir "artifacts/run_matrix/portable_${BUDGET}"' in text
    assert '--output-root "artifacts/runs/portable/${BUDGET}"' in text
    assert '--budget "${BUDGET}"' in text
    assert "maxionbench execute-run-matrix \\" in text
    assert '--matrix "artifacts/run_matrix/portable_${BUDGET}/run_matrix.json"' in text
    assert "maxionbench verify-promotion-gate \\" in text
    assert '--portable-results "artifacts/runs/portable/${BUDGET}"' in text
    assert "configs/scenarios_portable/s1_single_hop.yaml" in text
    assert "configs/scenarios_portable/s2_streaming_memory.yaml" in text
    assert "configs/scenarios_portable/s3_multi_hop.yaml" in text
    assert "maxionbench archive \\" in text
    assert "maxionbench services down" in text

    assert "run_docker_scenario.sh" not in text
    assert "run_workstation.sh" not in text
    assert "save_results_bundle.sh" not in text
