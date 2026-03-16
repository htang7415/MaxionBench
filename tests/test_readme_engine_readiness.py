from __future__ import annotations

from pathlib import Path


def test_readme_describes_study_scope_and_run_entrypoints() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "MaxionBench is a reproducible single-node benchmark study" in text
    assert "Qdrant, Milvus, Weaviate, OpenSearch k-NN, PostgreSQL + pgvector, LanceDB-service, LanceDB-inproc, FAISS CPU, FAISS GPU" in text
    assert "S1: ANN frontier" in text
    assert "S2: filtered ANN" in text
    assert "S3 / S3b: churn robustness" in text
    assert "S4: hybrid retrieval" in text
    assert "S5: rerank pipeline cost" in text
    assert "S6: fusion" in text
    assert "./run_slurm_pipeline.sh --cluster euler --container-image /shared/containers/maxionbench.sif" in text
    assert "./run_slurm_pipeline.sh --cluster nrel --container-image /shared/containers/maxionbench.sif" in text
    assert "gitignored local profile" in text
    assert "maxionbench/orchestration/slurm/profiles_local.yaml" in text

    assert "nawimem" not in text
    assert "pdelab" not in text
