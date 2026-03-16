from __future__ import annotations

from pathlib import Path


def test_readme_describes_study_scope_and_run_entrypoints() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "MaxionBench is a reproducible single-node benchmark study" in text
    assert "Qdrant, Milvus, Weaviate, OpenSearch k-NN, PostgreSQL + pgvector, LanceDB-service, LanceDB-inproc, FAISS CPU, FAISS GPU" in text
    assert "| Bundle | Contents | Role |" in text
    assert "| D1 | ann-benchmarks HDF5 | ANN microbench set |" in text
    assert "| D2 | BigANN 10M tier | large ANN anchor |" in text
    assert "| D3 | `yfcc-10M` | filtered ANN and churn robustness |" in text
    assert "| D4 | BEIR subsets plus CRAG slice | retrieval and RAG utility |" in text
    assert "| Scenario | Goal | Dataset |" in text
    assert "| S1 | ANN frontier | D1, D2 |" in text
    assert "| S2 | filtered ANN | D3 |" in text
    assert "| S3 | churn robustness | D3 |" in text
    assert "| S3b | bursty churn robustness | D3 |" in text
    assert "| S4 | hybrid retrieval | D4 |" in text
    assert "| S5 | rerank pipeline cost | D4 |" in text
    assert "| S6 | fusion | D4 |" in text
    assert "./run_slurm_pipeline.sh --slurm-profile <profile> --container-image /shared/containers/maxionbench.sif" in text
    assert "gitignored local profile" in text
    assert "maxionbench/orchestration/slurm/profiles_local.yaml" in text

    assert "Mac mini" not in text
    assert "mac mini" not in text
    assert "Euler" not in text
    assert "NREL" not in text
    assert "--cluster euler" not in text
    assert "--cluster nrel" not in text
    assert "nawimem" not in text
    assert "pdelab" not in text
