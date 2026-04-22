# MaxionBench

MaxionBench is a reproducible single-node benchmark study for retrieval infrastructure used in agentic applications.

## Benchmark study

The study reports matched-quality tradeoffs, p99 latency, throughput, robustness inflation, and RHU-normalized cost.

## Engines

| Engine | Category | Role in portable study | Notes |
| --- | --- | --- | --- |
| FAISS CPU | local baseline | exact/strong local baseline | paper matrix engine |
| LanceDB-inproc | embedded/local | upper-bound local reference | paper matrix engine |
| LanceDB-service | service wrapper | primary comparable LanceDB mode | paper matrix engine |
| PostgreSQL + pgvector | DB-first | service-backed portable engine | paper matrix engine |
| Qdrant | vector-first server | service-backed portable engine | paper matrix engine |

## Datasets

| Dataset | Source | Role in portable study | Notes |
| --- | --- | --- | --- |
| `scifact` | BEIR | S1 single-hop corpus | paper-path single-hop dataset |
| `fiqa` | BEIR | S1 single-hop corpus | paper-path single-hop dataset |
| `CRAG-500` | CRAG task 1/2 dev slice | S2 online event stream | one inserted supporting passage per event |
| `FRAMES-portable` | frozen local FRAMES + KILT preprocessing | S3 multi-hop evidence retrieval | one-time offline preprocessing artifact |

## Scenarios

| Scenario | Dataset | Goal | Concurrency pin | Pinned details |
| --- | --- | --- | --- | --- |
| S1 | `scifact`, `fiqa` | single-hop corpus retrieval | clients `{1, 4, 8}` | primary quality `nDCG@10` |
| S2 | `scifact` + `fiqa` background with `CRAG-500` events | streaming memory | read/write `8 / 2` | freshness probes at `T+1s` and `T+5s` |
| S3 | `FRAMES-portable` | multi-hop evidence retrieval | clients `{1, 4, 8}` | primary quality `evidence_coverage@10` |

## Run artifacts

Each run writes:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

## Operator Constraints

- The operator machine is an Apple Silicon Mac mini; it is the execution host, not part of the study storyline.
- The local operator workflow is controlled to fit within one day wall clock.
- `submit-portable` defaults to a 24-hour benchmark-execution deadline; lower `--deadline-hours` if setup, data, or embedding work consumes part of the day.
- GPU-required scenarios and distributed topologies are out of scope.
- Manual acquisition of FRAMES/KILT source inputs may happen before the one-day controlled workflow.

Portable figures are written to `artifacts/figures/final/`.

## How to run this benchmark study

Use the reduced local workflow.

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
python -m maxionbench.cli report --input artifacts/runs/portable --mode portable-agentic --out artifacts/figures/final
```
