# MaxionBench

MaxionBench is a reproducible single-node benchmark study for vector databases and retrieval infrastructure used in RAG systems.

## Benchmark study

- Engines: Qdrant, Milvus, Weaviate, OpenSearch k-NN, PostgreSQL + pgvector, LanceDB-service, LanceDB-inproc, FAISS CPU, FAISS GPU

The study reports matched-quality tradeoffs, p99 latency, throughput, robustness inflation, and RHU-normalized cost.

## Datasets

| Bundle | Contents | Role |
| --- | --- | --- |
| D1 | ann-benchmarks HDF5 | ANN microbench set |
| D2 | BigANN 10M tier | large ANN anchor |
| D3 | `yfcc-10M` | filtered ANN and churn robustness |
| D4 | BEIR subsets plus CRAG slice | retrieval and RAG utility |

## Scenarios

| Scenario | Goal | Dataset |
| --- | --- | --- |
| S1 | ANN frontier | D1, D2 |
| S2 | filtered ANN | D3 |
| S3 | churn robustness | D3 |
| S3b | bursty churn robustness | D3 |
| S4 | hybrid retrieval | D4 |
| S5 | rerank pipeline cost | D4 |
| S6 | fusion | D4 |

## Run artifacts

Each run writes:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

Figures are written to:

- `artifacts/figures/milestones/Mx/`
- `artifacts/figures/final/`

## How to run this benchmark study

### Local terminal workflow

Install, download, and preprocess:

```bash
python -m pip install -e ".[dev,engines,reporting]"
python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json
bash preprocess_all_datasets.sh
```

Then run the local end-to-end workflow in [command-mac.md](/Users/haotang/Library/CloudStorage/OneDrive-UW-Madison/MAX/Al/Research/MaxionBench/command-mac.md).

### Slurm cluster workflow

Use a prebuilt Apptainer image on shared storage, keep private cluster settings only in the gitignored local profile, and run:

```bash
./run_slurm_pipeline.sh --slurm-profile <profile> --container-image /shared/containers/maxionbench.sif
./run_slurm_pipeline.sh --slurm-profile <profile> --container-image /shared/containers/maxionbench.sif --launch
```

Detailed Slurm commands live in [command.md](/Users/haotang/Library/CloudStorage/OneDrive-UW-Madison/MAX/Al/Research/MaxionBench/command.md).

Private cluster account and partition values belong only in:

- `maxionbench/orchestration/slurm/profiles_local.yaml`

## Validate and generate figures

```bash
python -m maxionbench.cli validate --input artifacts/runs --strict-schema --json
python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1
python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final
```
