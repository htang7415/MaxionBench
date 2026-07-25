# MaxionBench

MaxionBench is a reproducible single-node decision-audit benchmark for retrieval infrastructure used as agentic operational memory.

## Benchmark study

The study reports conformance-gated workload quality, post-insert top-10 retrievability, max-row p99 latency, throughput, budget stability, paired audits, and a normalized context-cost proxy. Max-row p99 is the maximum archived p99 across configured B2 client rows and repeats for a candidate.

## Engines

| Engine | Category | Role in MaxionBench study | Notes |
| --- | --- | --- | --- |
| FAISS CPU | local baseline | exact/strong local baseline | included in the default benchmark matrix |
| LanceDB-inproc | embedded/local | upper-bound local reference | included in the default benchmark matrix |
| LanceDB-service | service wrapper | service-mode audit target | excluded by default because the archived local conformance table has no passing service row |
| PostgreSQL + pgvector | DB-first | service-backed MaxionBench engine | included in the default benchmark matrix |
| Qdrant | vector-first server | service-backed MaxionBench engine | included in the default benchmark matrix |

## Datasets

| Dataset | Source | Role in MaxionBench study | Notes |
| --- | --- | --- | --- |
| `scifact` | BEIR | S1 single-hop corpus | default single-hop dataset |
| `fiqa` | BEIR | S1 single-hop corpus | default single-hop dataset |
| `CRAG-500` | CRAG task 1/2 dev slice | S2 online event stream | one inserted supporting passage per event |
| `HotpotQA-MaxionBench` | frozen local HotpotQA dev distractor preprocessing | S3 multi-evidence retrieval | one-time offline preprocessing artifact |

## Scenarios

| Scenario | Dataset | Goal | Concurrency pin | Pinned details |
| --- | --- | --- | --- | --- |
| S1 | `scifact`, `fiqa` | single-hop corpus retrieval | clients `{1, 4, 8}` | primary quality `nDCG@10` |
| S2 | `scifact` + `fiqa` background with `CRAG-500` events | streaming memory | read/write `8 / 2` | post-insert top-10 retrievability probes at `T+1s` and `T+5s` |
| S3 | `HotpotQA-MaxionBench` | multi-evidence retrieval | clients `{1, 4, 8}` | primary quality `evidence_coverage@10` |

## Run artifacts

Each run writes:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

## Repository Layout

| Path | Purpose |
| --- | --- |
| `maxionbench/` | Benchmark package, engine adapters, orchestration, reports, runtime metadata, schemas, and CLI tools. |
| `configs/` | Pinned scenario, engine, and conformance configurations. |
| `docs/` | Public behavior cards, migration notes, and CI notes. |
| `tests/` | CI checks for configs, schemas, reports, workflows, and repository hygiene. |
| `dataset/processed/hotpot_portable/` | Frozen HotpotQA-MaxionBench fixture and checksums. |
| `artifacts/`, `results/`, `release/` | Local generated outputs; ignored by default and packaged explicitly when needed. |

Supporting documentation: [architecture](docs/architecture.md), [technical report](docs/technical_report.md), [contributing](docs/contributing.md), and [security](docs/security.md).

## Scope Constraints

- The local workflow is controlled to fit within one day wall clock on a single node.
- `submit` defaults to a 24-hour benchmark-execution deadline; lower `--deadline-hours` if setup, data, or embedding work consumes part of the day.
- GPU-required scenarios and distributed topologies are out of scope.
- The default S3 dataset is `HotpotQA-MaxionBench`, prepared from the official HotpotQA dev distractor release before timed execution.

MaxionBench figures are written to `artifacts/figures/final/`.

## Public Artifact Hygiene

The GitHub repository tracks source, configs, tests, lightweight dataset manifests, and public docs. Local benchmark outputs, release bundles, caches, and editor state are ignored by default so host paths, usernames, and machine-local metadata do not enter commits.

## How to run this benchmark study

Use the reduced local workflow.

Install the locked development environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
```

It covers:

- install + conformance
- dataset download + preprocessing
- embedding precompute
- Docker service startup
- MaxionBench `B0/B1/B2` matrix generation and execution
- reporting and archive commands

There are no required repo shell wrappers in the current workflow.

Primary workflow commands:

```bash
python -m maxionbench.cli workflow data --json
python -m maxionbench.cli submit --budget b0 --json
```

## Validate and generate figures

```bash
python -m maxionbench.cli validate --input artifacts/runs --strict-schema --json
python -m maxionbench.cli report \
  --input results/20260429T033427Z/runs \
  --mode maxionbench \
  --out artifacts/figures/final \
  --conformance-matrix artifacts/conformance/conformance_matrix.csv \
  --behavior-dir docs/behavior
```
