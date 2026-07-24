# MaxionBench Technical Report

Time: 2025.09 - 2026.02

## Purpose

This report summarizes the motivation, datasets, model, theory, methods, metrics, results, and discussion for MaxionBench. It is written as a technical project report rather than a file inventory, so implementation details are described by role instead of by source location.

## Executive Summary

MaxionBench is a reproducible single-node decision-audit benchmark for retrieval infrastructure used as agentic operational memory. Instead of producing a single leaderboard, it evaluates retrieval engines as deployment decision candidates under explicit conformance, quality, post-insert retrievability, latency, throughput, budget-stability, and normalized context-cost constraints.

The implemented benchmark includes:

- A Python benchmark package with CLI workflows, typed result schemas, engine adapters, dataset loaders, orchestration, reporting, and archive tooling.
- A shared adapter contract for lifecycle, CRUD, filtering, updates, deletes, query, batch query, flush/commit, compaction, and stats operations.
- Conformance gates that must pass before an engine is reportable.
- Three dataset-backed workloads covering BEIR-style static retrieval, CRAG-style streaming memory, and HotpotQA-derived multi-evidence retrieval.
- A B0/B1/B2 budget ladder with promotion gates and strict-schema validation.
- Generated paper-facing tables and figures from archived run artifacts.

The archived B0/B1/B2 snapshot validated 72 run directories with 0 strict-schema errors. Under the main 200 ms max-row p99 policy, the policy-selected strict-latency configuration was FAISS CPU with `BAAI/bge-small-en-v1.5` for all three workloads.

MaxionBench v0.1 is intentionally scoped to single-node local execution. It uses the CPU lane by default, Docker-backed local services for Qdrant and pgvector, in-process adapters for FAISS CPU and LanceDB-inproc, and no hidden retries during timed measurements. Distributed topologies, GPU-required scenarios, full-agent task success, full-Wikipedia runtime retrieval, and service-mode LanceDB paper results are outside this project version.

## Motivation

Agentic systems increasingly use retrieval backends as operational memory: they read from prior context, insert new evidence, and depend on retrieved passages to support future reasoning. A retrieval engine that looks strong on a static leaderboard can still be a poor deployment choice if it fails adapter correctness, has slow tail latency, does not expose newly inserted evidence quickly enough, or changes its recommendation under a realistic budget.

MaxionBench is motivated by that gap. It treats retrieval infrastructure selection as a deployment decision under constraints, not as a universal ranking problem. The benchmark asks whether a one-day local study can produce stable, auditable recommendations for agentic memory workloads while making quality, latency, freshness, cost proxy, and reproducibility tradeoffs visible.

## Model and Theory

MaxionBench models each reportable candidate as a concrete configuration of engine, embedding model, workload, client load, search parameters, and budget level. For a candidate `theta`, the benchmark records workload quality `Q(theta)`, performance `P(theta)`, post-insert retrievability `F(theta)` when relevant, resource behavior `R(theta)`, and normalized context-cost proxy `C(theta)`.

The theory behind the benchmark is constrained decision auditing. A candidate is considered only if it passes adapter conformance and reportability gates, satisfies the workload quality floor, and meets the selected p99 latency policy. Among feasible candidates, MaxionBench chooses the lowest context-cost configuration, breaking ties by lower tail latency and then higher throughput. This makes the benchmark objective-conditioned: different valid deployment policies can select different engines, so the result is a decision surface rather than a single global winner.

### Mathematical Definitions

Let `E` be the engine set, `M` the embedding-model set, `S` the workload set, `D_s` the dataset bundle for workload `s`, and `B` the budget set. A benchmark candidate is:

```text
theta = (e, m, s, D_s, b, u, k, lambda)
```

where `e in E` is the engine, `m in M` is the embedding model, `s in S` is the workload, `D_s` is the workload-specific corpus/query/qrel or event bundle, `b in B` is the budget level, `u` is the client-load setting, `k = 10` is the retrieval depth, and `lambda` contains engine search parameters such as IVF probes or HNSW ef.

For a workload `s`, budget `b`, and p99 policy threshold `tau`, the feasible decision set is:

```text
F(s, b, tau) = {
  theta :
    reportable(e) = 1,
    Q_s(theta) >= q_s_min,
    L99_max(theta) <= tau,
    F_s(theta) >= f_s_min when freshness is required
}
```

The selected strict-latency recommendation is the lexicographic minimum:

```text
theta_star(s, b, tau) =
  argmin_{theta in F(s,b,tau)}
    (C(theta), L99_max(theta), -T(theta), stable_sort_key(theta))
```

Here `C(theta)` is the normalized context-cost proxy, `L99_max(theta)` is the worst archived p99 row for the candidate, and `T(theta)` is throughput. The benchmark is therefore a constrained optimization problem, not a raw performance ranking.

## Benchmark System

### Benchmark Harness

The implemented system contains the following parts:

| Area | Role |
| --- | --- |
| Command workflow | Runs setup, data preparation, benchmark submission, validation, reporting, conformance checks, service lifecycle, and archive generation. |
| Engine adapters | Implements FAISS CPU, LanceDB in-process, LanceDB service, pgvector, Qdrant, and mock adapters behind one shared contract. |
| Adapter contract | Defines required request, result, statistic, lifecycle, data, retrieval, consistency, and control operations. |
| Conformance checks | Verifies required adapter behavior before an engine can be reported. |
| Scenarios | Implements portable static retrieval, streaming-memory retrieval, and multi-evidence retrieval workloads. |
| Orchestration | Expands benchmark configurations, runs scenarios, records metadata, and writes run artifacts. |
| Reporting | Generates result tables, decision surfaces, stability summaries, and figures from archived runs. |
| Validation tools | Check schemas, promotion gates, archive completeness, and release readiness. |

### Adapter Contract

Each reportable adapter implements:

- Lifecycle: `create`, `drop`, `reset`, `healthcheck`
- Data operations: `bulk_upsert`, `insert`, `update_vectors`, `update_payload`, `delete`
- Retrieval: `query`, `batch_query`
- Consistency/control: `flush_or_commit`, `set_index_params`, `set_search_params`, `optimize_or_compact`
- Stats: `vector_count`, `deleted_count`, `index_size_bytes`, `ram_usage_bytes`, `disk_usage_bytes`, `engine_uptime_s`

Conformance tests verify healthcheck, flush visibility, filter correctness, empty collection behavior, update semantics, delete semantics, batch query behavior, stats fields, compaction safety, and repeated flush stability.

### Engines Evaluated

| Engine | Role | Notes |
| --- | --- | --- |
| `faiss-cpu` | In-process CPU baseline | Reported rows use exact FlatIP; the recorded search sweep is ignored by the flat-index path. |
| `lancedb-inproc` | Embedded local reference | Uses local LanceDB table search without a secondary index; requires a local filesystem path with atomic `rename()` support. |
| `pgvector` | PostgreSQL + pgvector service | Reported runs use adapter-default IVF Flat search with `ivfflat_probes` sweep values `32` and `64`; conformance uses `index_method=none`. |
| `qdrant` | Qdrant service | Service profile starts local Qdrant; reported runs sweep `hnsw_ef` values `32` and `64`. |
| `lancedb-service` | Service wrapper / audit target | Available as an audit target, but excluded from the default paper matrix unless explicitly selected. |
| `mock` | Structural test adapter | Used for tests/conformance only; not a paper engine. |

The reportable archived engine set is `faiss-cpu`, `lancedb-inproc`, `pgvector`, and `qdrant`. The archived conformance matrix has five pass rows: `faiss-cpu`, `lancedb-inproc`, `mock`, `pgvector`, and `qdrant`. The support table marks `lancedb-service` as non-reportable because its behavior documentation exists but its conformance row is missing.

The local service profile starts Qdrant and pgvector containers. The project defaults are Qdrant `qdrant/qdrant:v1.17.1` and pgvector `pgvector/pgvector:0.8.2-pg16-trixie`.

## Methods

### Datasets

MaxionBench uses one dataset bundle per workload so that each decision is tied to a concrete retrieval setting rather than an abstract engine benchmark.

| Dataset | Used in | Role | Construction and scale |
| --- | --- | --- | --- |
| SciFact | S1, S2 | Scientific claim retrieval and part of the streaming-memory background. | Used in canonical BEIR corpus/query/qrel form. |
| FiQA | S1, S2 | Financial question-answer retrieval and part of the streaming-memory background. | Used in canonical BEIR corpus/query/qrel form. |
| CRAG-500 | S2 | Online event stream for post-insert retrievability. | Uses 500 CRAG task 1/2 development-slice events; each event contributes an inserted supporting passage, and CRAG is excluded from the static background so freshness is measured after insertion. |
| HotpotQA-MaxionBench | S3 | Multi-evidence retrieval under a top-10 context budget. | Frozen preprocessing of HotpotQA dev distractor; 66,635 documents, 7,405 questions, and 14,810 qrels. The retrieval unit is a context paragraph, and supporting facts are mapped to paragraph-level evidence documents. |

### Workloads

| Workload | Dataset(s) | Goal | Primary metric | Concurrency |
| --- | --- | --- | --- | --- |
| S1 single-hop | BEIR-format SciFact and FiQA | Static corpus retrieval | `nDCG@10` | read clients `{1, 4, 8}` |
| S2 streaming memory | 50K deterministic SciFact/FiQA background plus CRAG-500 events | Static retrieval plus post-insert top-10 retrievability | `nDCG@10` plus post-insert hit metrics | read/write clients `8 / 2` |
| S3 multi-hop | HotpotQA-MaxionBench from HotpotQA dev distractor | Multi-evidence retrieval under a top-10 context budget | `evidence_coverage@10` | read clients `{1, 4, 8}` |

HotpotQA-MaxionBench is a frozen preprocessing of the official HotpotQA dev distractor split. The archived manifest records 66,635 documents, 7,405 questions, and 14,810 qrels.

### Embedding Models

The paper-path matrix uses two local embedding tiers:

- `BAAI/bge-small-en-v1.5`, 384 dimensions
- `BAAI/bge-base-en-v1.5`, 768 dimensions

Embeddings are precomputed before timed query measurement. Latency measures `adapter.query` plus top-k materialization, including local service/container overhead inside the adapter call when applicable. Embedding choice is a first-class matrix axis and affects quality, latency, memory footprint, index size, and `task_cost_est`.

### Budget Ladder

| Budget | Warmup | Measurement | Repeats |
| --- | ---: | ---: | ---: |
| B0 | 10 s | 10 s | 1 |
| B1 | 15 s | 30 s | 1 |
| B2 | 30 s | 60 s | 2 |

Promotion gates prevent weak configurations from advancing. Reportable quality floors are `nDCG@10 >= 0.25` for S1/S2 and `evidence_coverage@10 >= 0.30` for S3. S2 also requires post-insert retrievability floors during promotion.

Common scenario pins are profile `maxionbench`, seed `42`, top-k `10`, retries disabled, `c_llm_in = 0.15`, and search sweep values appropriate to each adapter.

### Decision Rule

The strict-latency report rule:

1. Require conformance pass and behavior-card coverage.
2. Require the workload quality floor.
3. Require max-row p99 below the stated threshold. The main policy uses 200 ms.
4. Rank surviving configurations by `task_cost_est`.
5. Break ties by lower max-row p99, then higher throughput.

The context-cost proxy is:

```text
C(theta) = C_retrieval(theta) + C_embedding(theta) + C_context(theta)
C_context(theta) = c_llm_in * avg_retrieved_input_tokens(theta)
```

For the archived run, offline embeddings are precomputed, so `C_embedding(theta) = 0`, and `c_llm_in = 0.15`. This is a normalized context-cost proxy, not a cloud-dollar estimate. Max-row p99 is:

```text
L99_max(theta) = max_{r in rows(theta)} p99_ms(r)
```

where `rows(theta)` contains the configured client rows and repeats for the candidate at the selected budget.

## Metrics

MaxionBench reports correctness, quality, freshness, performance, cost, and stability metrics together because no single metric is enough for agentic retrieval infrastructure selection.

| Metric family | Metrics | Meaning |
| --- | --- | --- |
| Conformance | adapter behavior pass/fail | Whether an engine satisfies the shared lifecycle, CRUD, query, filtering, update, delete, flush, batch, compaction, and stats contract before benchmarking. |
| Retrieval quality | `nDCG@10`, `evidence_coverage@10` | Static single-hop and streaming quality use `nDCG@10`; multi-evidence retrieval uses the fraction of required evidence recovered in the top-10 context. |
| Freshness | `post_insert_hit@10` at short delays | Whether newly inserted evidence becomes retrievable in top-10 results after acknowledged insertion. |
| Latency and throughput | p50, p95, p99, max-row p99, qps | Tail latency and throughput under configured client loads; the main policy constrains max-row p99. |
| Cost proxy | `task_cost_est` | A normalized retrieved-context burden combining retrieval, embedding, and estimated LLM input-context cost. |
| Budget stability | Spearman rank correlation, top-1 agreement, top-2 agreement | Whether short screening budgets preserve the decisions observed at the final budget level. |
| Uncertainty and audits | confidence intervals, paired deltas, matched-query checks | Whether apparent aggregate differences survive closer comparison under matched queries or repeat runs. |

### Key Metric Formulae

For single-hop and streaming retrieval, the quality metric is `nDCG@10`:

```text
DCG@10(q) = sum_{i=1}^{10} (2^{rel_i(q)} - 1) / log2(i + 1)
nDCG@10(q) = DCG@10(q) / IDCG@10(q)
nDCG@10 = mean_q nDCG@10(q)
```

For multi-evidence retrieval, the quality metric is evidence coverage:

```text
evidence_coverage@10(q) =
  | retrieved_docs@10(q) intersect gold_evidence(q) | / | gold_evidence(q) |

evidence_coverage@10 = mean_q evidence_coverage@10(q)
```

For streaming memory, post-insert retrievability at delay `delta` is:

```text
post_insert_hit@10,delta =
  mean_i 1{ inserted_evidence_i in retrieved_docs@10(query_i, t_i + delta) }
```

Throughput and budget stability are summarized as:

```text
qps = completed_queries / measurement_seconds
rho_B0_B2 = SpearmanRankCorr(rank_B0, rank_B2)
top_k_agreement = |TopK_B0 intersect TopK_B2| / k
```

## Main Results

All result tables use the same dataset mapping: S1 is SciFact plus FiQA static retrieval, S2 is a deterministic SciFact/FiQA background with CRAG-500 streaming events, and S3 is HotpotQA-MaxionBench multi-evidence retrieval.

### Strict-Latency Winners at 200 ms Max-Row p99

| Workload | Selected engine / embedding | Quality result | Post-insert result | p99 mean / max | Task cost | Evidence size |
| --- | --- | --- | --- | --- | ---: | --- |
| S1 single-hop | FAISS CPU / bge-small | `nDCG@10 = 0.5055`, 95% CI `0.4942-0.5163` | n/a | `6.3 ms / 11.2 ms` | `262.759` | 5,364 query observations |
| S2 streaming memory | FAISS CPU / bge-small | `nDCG@10 = 0.5055`, 95% CI `0.4870-0.5237` | `post_insert_hit@10,5s = 0.830`, 95% CI `0.8055-0.8520` | `10.0 ms / 10.0 ms` | `262.759` | 1,788 query observations, 1,000 post-insert events |
| S3 multi-hop | FAISS CPU / bge-small | `evidence_coverage@10 = 0.8515`, 95% CI `0.8489-0.8543` | n/a | `11.6 ms / 22.8 ms` | `129.325` | 30,000 query observations |

Interpretation:

- FAISS CPU with bge-small is the selected strict-latency row for S1, S2, and S3 under the local CPU run profile.
- S1 and S2 are close cost/quality ties against LanceDB in-process, resolved by lower tail latency.
- S3 has a clearer strict-latency margin: the nearest strict-cost alternative, Qdrant with bge-small, is lower quality and slightly more expensive, although its p99 is slightly lower.

### Objective Sensitivity

MaxionBench reports multiple objectives because plausible deployment policies choose different configurations.

| Workload | Strict 200 ms winner | Cost-only / no-p99 winner | Quality-first winner | Quality-first metric |
| --- | --- | --- | --- | --- |
| S1 single-hop | FAISS CPU / bge-small | FAISS CPU / bge-small | LanceDB in-process / bge-base | `nDCG@10 = 0.5268`, p99 max `337.4 ms` |
| S2 streaming memory | FAISS CPU / bge-small | FAISS CPU / bge-small | FAISS CPU / bge-base | `nDCG@10 = 0.5268`, p99 max `21.7 ms` |
| S3 multi-hop | FAISS CPU / bge-small | LanceDB in-process / bge-small | pgvector / bge-base | `evidence_coverage@10 = 0.8726`, p99 max `1217.3 ms` |

The S3 aggregate quality-first pgvector result is not treated as a strong pgvector advantage. A 5,000-query matched audit showed pgvector IVF32 and IVF64 each at mean delta `-0.0001` evidence coverage versus FAISS exact FlatIP, with 95% intervals including zero.

### Threshold Sensitivity

The p99 policy threshold changes only S3 in the archived threshold sweep.

| p99 policy | S1 winner | S2 winner | S3 winner |
| --- | --- | --- | --- |
| 100 ms | FAISS CPU / bge-small | FAISS CPU / bge-small | FAISS CPU / bge-small |
| 200 ms | FAISS CPU / bge-small | FAISS CPU / bge-small | FAISS CPU / bge-small |
| 500 ms | FAISS CPU / bge-small | FAISS CPU / bge-small | pgvector / bge-small |
| No p99 cap | FAISS CPU / bge-small | FAISS CPU / bge-small | LanceDB in-process / bge-small |

This is why the project frames results as objective-conditioned decision audits instead of one global engine ranking.

### Decision Margins

| Workload | Closest strict-cost competitor | Quality delta vs selected | Cost delta vs selected | Selected p99 max | Competitor p99 max | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S1 single-hop | LanceDB in-process / bge-small | `0.0000` | approximately `0.0000` | `11.2 ms` | `172.1 ms` | Cost/quality tie resolved by p99. |
| S2 streaming memory | LanceDB in-process / bge-small | `0.0000` | approximately `0.0000` | `10.0 ms` | `147.8 ms` | Cost/quality tie resolved by p99. |
| S3 multi-hop | Qdrant / bge-small | `-0.0364` | `+0.5102` | `22.8 ms` | `18.3 ms` | Competitor is faster but lower quality and more expensive. |

### Budget Stability

| Workload | B0 -> B2 Spearman rho | B0 -> B2 top-1 agreement | B0 -> B2 top-2 agreement | Interpretation |
| --- | ---: | ---: | ---: | --- |
| S1 single-hop | `0.786` | `0.0` | `1.0` | Screening did not preserve the final top-1 choice. |
| S2 streaming memory | `0.429` | `0.0` | `1.0` | Screening did not preserve the final top-1 choice. |
| S3 multi-hop | `0.310` | `1.0` | `1.0` | Top-1 was stable even though lower-rank ordering was noisy. |

The budget analysis supports reporting the budget level with each decision rather than assuming a short screening run is enough.

## Targeted Audits and Stress Checks

### S3 Matched Quality Audit

The S3 quality-first claim was audited with a 5,000-query matched subset:

- FAISS exact FlatIP / bge-base: `evidence_coverage@10 = 0.8712`, p99 `17.6 ms`.
- pgvector IVF32 / bge-base: `evidence_coverage@10 = 0.8711`, p99 `306.0 ms`.
- pgvector IVF64 / bge-base: `evidence_coverage@10 = 0.8711`, p99 `346.2 ms`.
- Paired pgvector-minus-FAISS mean delta for both pgvector settings: `-0.0001`, with 95% interval approximately `[-0.0003, 0.0]`.

Conclusion: the apparent aggregate pgvector S3 quality edge is not a detectable matched-query advantage at this sample size.

### S2 FAISS vs Qdrant Same-Machine Check

A larger same-machine S2 check compared FAISS CPU and Qdrant on bge-small with two B2 repeats:

- Both engines reached `post_insert_hit@10,1s = 0.84` and `post_insert_hit@10,5s = 0.84` over 200 post-insert observations.
- Qdrant-minus-FAISS `nDCG@10` mean delta was `-0.00267`, with 95% CI `[-0.00521, -0.00058]`.
- Qdrant had lower latency in this bounded check, but no quality or post-insert retrievability advantage.

Conclusion: the check does not overturn the strict FAISS decision because the objective includes quality, cost, and p99 policy together.

### Strict FAISS Repeat Audit

Same-machine repeat checks reproduced the strict FAISS CPU / bge-small quality values and stayed below the 200 ms policy:

| Workload | Repeat rows | Quality mean | p99 min / median / max |
| --- | ---: | ---: | --- |
| S1 | 9 | `nDCG@10 = 0.5055` | `2.4 / 8.9 / 27.7 ms` |
| S2 | 2 | `nDCG@10 = 0.5055`, post-insert@5s `0.84` | `21.4 / 47.8 / 74.1 ms` |
| S3 | 9 | `evidence_coverage@10 = 0.8515` | `3.6 / 9.6 / 20.6 ms` |

### Vector-Scale Sanity Check

A controlled vector-only Qdrant HNSW stress check used real HotpotQA-MaxionBench bge-small vectors plus deterministic random distractors:

- FAISS FlatIP p99 was `76.73 ms` at 1,000,000 vectors; FAISS IVF p99 was `9.45 ms` with recall@10 `0.9994` versus FlatIP.
- At 1,000,000 vectors, Qdrant HNSW reached recall@10 `0.9994` versus exact FAISS FlatIP.
- Qdrant p99 was `10.05 ms` for 512 queries.
- This is not a full MaxionBench workload row, but it shows why the local 50K-66K exact-baseline ordering should not be overgeneralized to larger production-scale indices.

## Discussion

The main result is that FAISS CPU with bge-small is the best strict-latency local decision under the 200 ms max-row p99 policy for all three workloads. That result should be read as a policy-conditioned recommendation for the measured single-node CPU setting, not as a universal claim that FAISS is always the best retrieval infrastructure.

The sensitivity tables show why MaxionBench reports decision surfaces instead of a single leaderboard. If the deployment policy removes the p99 cap or prioritizes raw quality, LanceDB in-process, FAISS bge-base, or pgvector can become selected under some workloads. The benchmark therefore makes the decision objective explicit and exposes where a recommendation is stable or fragile.

The targeted audits also narrow the claims. The S3 matched-query audit shows that pgvector's aggregate quality-first advantage does not survive a paired comparison against FAISS exact FlatIP at the audited sample size. The S2 same-machine check shows Qdrant can be faster in a bounded run, but without enough quality or freshness advantage to overturn the strict objective. The vector-scale sanity check shows that larger corpus sizes can favor indexed service engines, so the local 50K-66K corpus result should not be extrapolated without scale-specific evaluation.

## Reproducibility and Validation

Archived evidence:

- Runs directory copied into archive: 972 files
- Generated figures/tables copied into archive: 48 files
- HotpotQA-MaxionBench artifact copied into archive: 27 files
- Conformance artifacts copied into archive: 27 files
- Strict-schema release validation: 72 run directories checked, 0 errors
- Dataset provenance records pin the BEIR subsets, CRAG-500 event slice, HotpotQA-MaxionBench counts, dataset manifests, and checksums used by the reported runs.

Recorded runtime profile for the archived strict FAISS run:

- CPU-only local host
- Apple M4, 10 logical CPU cores, 16 GB RAM
- Python 3.11.14
- Docker 29.4.0
- No GPU rows in the reported matrix

Each run records structured measurements, runtime metadata, resolved configuration, status, logs, and per-observation records. The archived snapshot validates that the reported numbers can be traced back to strict schemas, dataset manifests, checksums, and generated tables without relying on hidden manual edits.

## Resume Reference Bullets

Use or adapt these depending on the target role:

- Built MaxionBench, a conformance-gated Python benchmark for evaluating vector retrieval infrastructure as operational memory for agentic AI systems.
- Designed a shared adapter contract for FAISS, LanceDB, pgvector, and Qdrant covering lifecycle, CRUD, filtering, update/delete semantics, batch query, flush/commit, compaction, and resource stats.
- Implemented B0/B1/B2 benchmark orchestration with promotion gates, strict result schemas, reproducible run metadata, and archive generation; validated 72 archived run directories with 0 schema errors.
- Built retrieval workloads for BEIR-style single-hop retrieval, CRAG-style streaming memory with post-insert top-10 probes, and HotpotQA-derived multi-evidence retrieval over 66,635 documents and 7,405 questions.
- Added decision-audit reporting that separates strict-latency, cost-only, and quality-first objectives and reports quality, max-row p99, context-cost proxy, post-insert retrievability, budget stability, and paired margins.
- Demonstrated that under a 200 ms max-row p99 policy, FAISS CPU with bge-small was the selected local strict-latency configuration across S1, S2, and S3, with S3 reaching `evidence_coverage@10 = 0.8515` and max-row p99 `22.8 ms`.
- Ran paired and targeted audits showing that aggregate quality differences can disappear under matched-query analysis, including a 5,000-query S3 audit where pgvector showed no detectable evidence-coverage advantage over FAISS exact FlatIP.

## Interview Talking Points

- The project is about decision quality, not claiming a universal best vector database.
- The key engineering choice was to gate every reportable engine through the same adapter contract and conformance tests before measuring performance.
- The benchmark measures post-insert top-k retrievability for streaming memory, which is more agent-facing than a low-level index visibility check.
- The result tables intentionally expose objective sensitivity: changing the p99 policy or optimizing quality alone can change the selected engine.
- The archived results are reproducible through explicit run artifacts, hardware/runtime metadata, schemas, dataset manifests, checksums, and report-generation code.

## Limitations and Scope

- Results are from a single CPU-only local runtime profile and should not be interpreted as production-cluster rankings.
- FAISS CPU exact FlatIP is especially competitive at the 50K-66K local corpus scale used here; larger vector-scale checks show that service HNSW engines can become more favorable at larger corpus sizes.
- `task_cost_est` is a normalized retrieved-context burden proxy, not a cloud-dollar estimate.
- S2 post-insert metrics measure whether inserted evidence appears in top-10 retrieval results after ACK, not low-level index visibility.
- Offline embedding is excluded from latency and included only through the context-cost accounting path.
