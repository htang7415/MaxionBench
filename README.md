# MaxionBench

MaxionBench is a reproducible single-node benchmark study for vector databases and retrieval infrastructure used in RAG systems.

## What this benchmark study covers

- Engines: Qdrant, Milvus, Weaviate, OpenSearch k-NN, PostgreSQL + pgvector, LanceDB-service, LanceDB-inproc, FAISS CPU, FAISS GPU
- Datasets:
  - D1: ann-benchmarks HDF5
  - D2: BigANN 10M tier
  - D3: `yfcc-10M`
  - D4: BEIR subsets plus the CRAG slice
- Scenarios:
  - S1: ANN frontier
  - S2: filtered ANN
  - S3 / S3b: churn robustness
  - S4: hybrid retrieval
  - S5: rerank pipeline cost
  - S6: fusion

The study reports matched-quality tradeoffs, p99 latency, throughput, robustness inflation, and RHU-normalized cost.

## Run artifacts

Each run writes:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

Figures are written to:

- `artifacts/figures/milestones/Mx/`
- `artifacts/figures/final/`

## Source of truth

Use these files in order:

1. `project.md`
2. `prompt.md`
3. `document.md`
4. `command.md`
5. `command-mac.md`

## How to run this benchmark study

### Local terminal workflow on Mac mini M4

Install, download, and preprocess:

```bash
python -m pip install -e ".[dev,engines,reporting]"
python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json
bash preprocess_all_datasets.sh
```

Then follow the local end-to-end commands in [command-mac.md](/Users/haotang/Library/CloudStorage/OneDrive-UW-Madison/MAX/Al/Research/MaxionBench/command-mac.md).

### Slurm workflow on Euler or NREL

Use a prebuilt Apptainer image on shared storage, keep private cluster settings only in the gitignored local profile, and run:

```bash
./run_slurm_pipeline.sh --cluster euler --container-image /shared/containers/maxionbench.sif
./run_slurm_pipeline.sh --cluster euler --container-image /shared/containers/maxionbench.sif --launch
```

or

```bash
./run_slurm_pipeline.sh --cluster nrel --container-image /shared/containers/maxionbench.sif
./run_slurm_pipeline.sh --cluster nrel --container-image /shared/containers/maxionbench.sif --launch
```

Full Slurm commands live in [command.md](/Users/haotang/Library/CloudStorage/OneDrive-UW-Madison/MAX/Al/Research/MaxionBench/command.md).

Private cluster account and partition values belong only in:

- `maxionbench/orchestration/slurm/profiles_local.yaml`

## Validate and generate figures

```bash
python -m maxionbench.cli validate --input artifacts/runs --strict-schema --json
python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1
python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final
```
