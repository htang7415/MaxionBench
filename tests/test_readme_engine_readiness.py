from __future__ import annotations

from pathlib import Path


def test_readme_describes_study_scope_and_run_entrypoints() -> None:
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
    assert "| S2 | D3 | filtered ANN selectivity sweep | clients `32` | selectivity `0.1% / 1% / 10% / 50%`, p99 inflation vs unfiltered anchor |" in text
    assert "| S3 | D3 | dynamic churn, smooth | read/write `32 / 8` | open-loop `1000 req/s`: read `800`, insert `100`, update `50`, delete `50`; maintenance every `60s` |" in text
    assert "| S3b | D3 | dynamic churn, bursty | read/write `32 / 8` | ON `30s`, OFF `90s`; ON writes `8x`, OFF writes `0.25x`; maintenance every `60s` |" in text
    assert "| S4 | D4 | hybrid retrieval | clients `16` | dense vs BM25+dense, RRF `k=60`, candidate budget `200/200` |" in text
    assert "| S5 | D4 | candidate generation + rerank | clients `16` | candidate budgets `{50, 200, 1000}`; reranker `BAAI/bge-reranker-base`, `max_seq_len=512`, `fp16`, `batch_size=32` |" in text
    assert "| S6 | D4 | multi-index fusion | clients `16` | fusion budget `200/200`, RRF `k=60`; first deferrable scenario if schedule risk appears |" in text
    assert "Keep cluster-local defaults in gitignored local files such as `.env.slurm.nrel`, `.env.slurm.euler`, and `maxionbench/orchestration/slurm/profiles_local.yaml`." in text
    assert "The wrapper auto-loads and refreshes `.env.slurm.<cluster>` for the current run" in text
    assert "bash run_slurm_pipeline.sh --cluster nrel" in text
    assert "Dry-run prints the submit plan only." in text
    assert "`--launch` prepares the shared directory tree and builds any missing `.sif` images under `${MAXIONBENCH_SHARED_ROOT}/containers/` before submitting jobs." in text
    assert "Copied example values such as `your-account`, `YOUR_PRIVATE_PARTITION`, or `/shared/containers/...` are rejected before submission." in text
    assert "Large Apptainer build cache/tmp data defaults to `${MAXIONBENCH_SHARED_ROOT}/.cache/apptainer`" in text
    assert "By default, dataset, cache, result, figure, and Hugging Face cache paths are derived from the repository root that contains `run_slurm_pipeline.sh`." in text
    assert "--shared-root /shared/path/maxionbench" in text

    assert "Mac mini" not in text
    assert "mac mini" not in text
    assert "--cluster euler" not in text
    assert "nawimem" not in text
    assert "pdelab" not in text
