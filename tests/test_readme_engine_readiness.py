from __future__ import annotations

from pathlib import Path


def test_readme_describes_study_scope_and_workstation_entrypoints() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "MaxionBench is a reproducible single-node benchmark study" in text
    assert "| Engine | Category | Primary role in study | Notes |" in text
    assert "| Qdrant | vector-first server | ANN, filtered ANN, churn | networked engine |" in text
    assert "| Milvus | vector-first server | ANN, filtered ANN, churn, GPU-index track if available | networked engine |" in text
    assert "| OpenSearch k-NN | hybrid/search-first | ANN and hybrid retrieval workflows | networked engine |" in text
    assert "| PostgreSQL + pgvector | DB-first | ANN baseline in relational setting | networked engine |" in text
    assert "| LanceDB-service | service wrapper | comparable service-mode LanceDB result | primary LanceDB comparison mode |" in text
    assert "| FAISS GPU | baseline | GPU ANN baseline | only when GPU is available |" in text
    assert "| Bundle | Source | Contents | Default scale | Study role | Notes |" in text
    assert "| D1 | ann-benchmarks HDF5 | `glove-100-angular`, `sift-128-euclidean`, `gist-960-euclidean` | benchmark-native | ANN microbench set | uses benchmark-provided or exact FAISS Flat ground truth |" in text
    assert "| D2 | ann-benchmarks HDF5 | `deep-image-96-angular` | 10M | large ANN anchor | pinned large-scale ANN dataset for v0.1 |" in text
    assert "| D3 | `big-ann-benchmarks` | `yfcc-10M` | 10M | filtered ANN and churn robustness | preserves official filtered-query semantics and is normalized under `dataset/D3/yfcc-10M/`; 50M is optional only if scratch preflight passes |" in text
    assert "| D4 | BEIR + CRAG | BEIR: `scifact`, `fiqa`, `nfcorpus`; CRAG: `data/crag_task_1_and_2_dev_v4.jsonl.bz2` with local slice `crag_task_1_and_2_dev_v4.first_500.jsonl` | pinned local bundle | retrieval and RAG utility | BEIR uses official qrels; CRAG slice is treated as weak-label / pipeline-realism data |" in text
    assert "| Scenario | Dataset | Goal | Concurrency pin | Pinned details |" in text
    assert "| S1 | D1, D2 | ANN frontier | clients `{1, 8, 32, 64}` | matched-quality Pareto frontier, build/load time, footprint |" in text
    assert "| S5 | D4 | candidate generation + rerank | clients `16` | candidate budgets `{50, 200, 1000}`; reranker `BAAI/bge-reranker-base`, `max_seq_len=512`, `fp16`, `batch_size=32` |" in text
    assert "Linux x86_64 workstation is the primary full-run host." in text
    assert "Mac is the reduced local/dev host." in text
    assert "Default full-run lane is CPU-only; GPU scenarios are explicit opt-in because the A100 may be shared." in text
    assert "Use the direct terminal workflow in `command.md`." in text
    assert "There are no required repo shell wrappers in the current workflow." in text

    assert ".env.slurm" not in text
    assert "run_slurm_pipeline.sh" not in text
    assert "Apptainer" not in text
    assert "run_workstation.sh" not in text
    assert "preprocess_all_datasets.sh" not in text
