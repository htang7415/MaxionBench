# MaxionBench

MaxionBench is a reproducible single-node decision-audit benchmark for retrieval infrastructure used as agentic operational memory.

## Benchmark study

The study reports conformance-gated workload quality, post-insert top-10 retrievability, max-row p99 latency, throughput, budget stability, paired audits, and a normalized context-cost proxy. Max-row p99 is the maximum archived p99 across configured B2 client rows and repeats for a candidate.

## Engines

| Engine | Category | Role in MaxionBench study | Notes |
| --- | --- | --- | --- |
| FAISS CPU | local baseline | exact/strong local baseline | reported paper engine |
| LanceDB-inproc | embedded/local | upper-bound local reference | reported paper engine |
| LanceDB-service | service wrapper | service-mode audit target | excluded from the paper matrix because the archived local conformance table has no passing service row |
| PostgreSQL + pgvector | DB-first | service-backed MaxionBench engine | reported paper engine |
| Qdrant | vector-first server | service-backed MaxionBench engine | reported paper engine |

## Datasets

| Dataset | Source | Role in MaxionBench study | Notes |
| --- | --- | --- | --- |
| `scifact` | BEIR | S1 single-hop corpus | paper-path single-hop dataset |
| `fiqa` | BEIR | S1 single-hop corpus | paper-path single-hop dataset |
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
| `docs/` | Public behavior cards, migration notes, CI notes, and NeurIPS artifact inventory. |
| `submission/` | Public NeurIPS submission handoff bundle: PDF/source, figures, tables, metadata, and submission-facing docs. |
| `tests/` | CI checks for configs, schemas, reports, workflows, and repository hygiene. |
| `dataset/processed/hotpot_portable/` | Lightweight tracked HotpotQA-MaxionBench manifest and checksum fixtures. |
| `artifacts/`, `results/`, `paper/`, `release/` | Local generated outputs; ignored by default and packaged explicitly when needed. |

## Scope Constraints

- The local workflow is controlled to fit within one day wall clock on a single node.
- `submit` defaults to a 24-hour benchmark-execution deadline; lower `--deadline-hours` if setup, data, or embedding work consumes part of the day.
- GPU-required scenarios and distributed topologies are out of scope.
- The primary S3 paper path is `HotpotQA-MaxionBench`, prepared from the official HotpotQA dev distractor release before timed execution.

MaxionBench figures are written to `artifacts/figures/final/`.

## Public Artifact Hygiene

The GitHub repository tracks source, configs, tests, lightweight dataset manifests, and public docs. Local benchmark outputs, paper build products, release bundles, caches, editor state, and reviewer-package staging directories are ignored by default so host paths, usernames, and machine-local metadata do not enter commits.

NeurIPS-facing public documents that should be visible on GitHub are mirrored under `submission/`. The ignored `paper/` tree remains a local staging area and should not be assumed to be part of a normal push.

## How to run this benchmark study

Use the reduced local workflow.

It covers:

- install + conformance
- dataset download + preprocessing
- embedding precompute
- Docker service startup
- MaxionBench `B0/B1/B2` matrix generation and execution
- reporting and archive commands

There are no required repo shell wrappers in the current workflow.

Primary paper-path commands:

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
