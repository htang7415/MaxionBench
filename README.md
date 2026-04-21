# MaxionBench

MaxionBench is a reproducible single-node benchmark study for vector databases and retrieval infrastructure used in RAG systems.

## Benchmark study

The study reports matched-quality tradeoffs, p99 latency, throughput, robustness inflation, and RHU-normalized cost.

## Engines

| Engine | Category | Primary role in study | Notes |
| --- | --- | --- | --- |
| Qdrant | vector-first server | ANN, filtered ANN, churn | networked engine |
| Milvus | vector-first server | ANN, filtered ANN, churn, GPU-index track if available | networked engine |
| Weaviate | hybrid/search-first | ANN and retrieval workflows | networked engine |
| OpenSearch k-NN | hybrid/search-first | ANN and hybrid retrieval workflows | networked engine |
| PostgreSQL + pgvector | DB-first | ANN baseline in relational setting | networked engine |
| LanceDB-service | service wrapper | comparable service-mode LanceDB result | primary LanceDB comparison mode |
| LanceDB-inproc | embedded/local | upper-bound local LanceDB result | secondary reference mode |
| FAISS CPU | baseline | exact/strong ANN baseline | also used for exact ground truth where needed |
| FAISS GPU | baseline | GPU ANN baseline | only when GPU is available |

## Datasets

| Bundle | Source | Contents | Default scale | Study role | Notes |
| --- | --- | --- | --- | --- | --- |
| D1 | ann-benchmarks HDF5 | `glove-100-angular`, `sift-128-euclidean`, `gist-960-euclidean` | benchmark-native | ANN microbench set | uses benchmark-provided or exact FAISS Flat ground truth |
| D2 | ann-benchmarks HDF5 | `deep-image-96-angular` | 10M | large ANN anchor | pinned large-scale ANN dataset for v0.1 |
| D3 | `big-ann-benchmarks` | `yfcc-10M` | 10M | filtered ANN and churn robustness | preserves official filtered-query semantics and is normalized under `dataset/D3/yfcc-10M/`; 50M is optional only if scratch preflight passes |
| D4 | BEIR + CRAG | BEIR: `scifact`, `fiqa`, `nfcorpus`; CRAG: `data/crag_task_1_and_2_dev_v4.jsonl.bz2` with local slice `crag_task_1_and_2_dev_v4.first_500.jsonl` | pinned local bundle | retrieval and RAG utility | BEIR uses official qrels; CRAG slice is treated as weak-label / pipeline-realism data |

## Scenarios

| Scenario | Dataset | Goal | Concurrency pin | Pinned details |
| --- | --- | --- | --- | --- |
| S1 | D1, D2 | ANN frontier | clients `{1, 8, 32, 64}` | matched-quality Pareto frontier, build/load time, footprint |
| S2 | D3 | filtered ANN selectivity sweep | clients `32` | selectivity `0.1% / 1% / 10% / 50%`, p99 inflation vs unfiltered anchor |
| S3 | D3 | dynamic churn, smooth | read/write `32 / 8` | open-loop `1000 req/s`: read `800`, insert `100`, update `50`, delete `50`; maintenance every `60s` |
| S3b | D3 | dynamic churn, bursty | read/write `32 / 8` | ON `30s`, OFF `90s`; ON writes `8x`, OFF writes `0.25x`; maintenance every `60s` |
| S4 | D4 | hybrid retrieval | clients `16` | dense vs BM25+dense, RRF `k=60`, candidate budget `200/200` |
| S5 | D4 | candidate generation + rerank | clients `16` | candidate budgets `{50, 200, 1000}`; reranker `BAAI/bge-reranker-base`, `max_seq_len=512`, `fp16`, `batch_size=32` |
| S6 | D4 | multi-index fusion | clients `16` | fusion budget `200/200`, RRF `k=60`; first deferrable scenario if schedule risk appears |

## Run artifacts

Each run writes:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

## Workstation targets

- Linux x86_64 workstation is the primary full-run host.
- Mac is the reduced local/dev host.
- Default full-run lane is CPU-only; GPU scenarios are explicit opt-in because the A100 may be shared.
- The Mac lane does not claim support for full D2/D3 paper-scale runs or the CUDA-enforced S5 / GPU tracks.

Figures are written to:

- `artifacts/figures/milestones/Mx/`
- `artifacts/figures/final/`

## How to run this benchmark study

Use the reduced local workflow in `command-mac.md`.

It covers:

- install + conformance
- dataset download + preprocessing
- embedding precompute
- Docker service startup
- portable `B0/B1/B2` matrix generation and execution
- reporting and archive commands

There are no required repo shell wrappers in the current workflow.

## Validate and generate figures

```bash
python -m maxionbench.cli validate --input artifacts/runs --strict-schema --json
python -m maxionbench.cli report --input artifacts/runs --mode milestones --milestone-id M1
python -m maxionbench.cli report --input artifacts/runs --mode final --out artifacts/figures/final
```
