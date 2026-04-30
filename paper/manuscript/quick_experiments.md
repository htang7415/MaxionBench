# Quick experiment and validation notes

These checks were run during manuscript preparation to improve NeurIPS readiness without replacing the archived paper bundle.

## Artifact validation

Command:

```bash
python -m maxionbench.cli validate --input artifacts/runs/portable --strict-schema --json
```

Result: pass, 72 run directories checked, 0 errors.

## Report regeneration

Command:

```bash
python -m maxionbench.cli report --input artifacts/runs/portable --mode portable-agentic --out /tmp/maxionbench_neurips_report_check --conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior
```

Result: completed successfully and produced the expected paper-facing CSV, TeX, PDF, PNG, and metadata files. `diff -q` confirmed that regenerated `neurips_main_results.csv`, `portable_decision_table.csv`, `portable_support_table.csv`, and `portable_stability.csv` match the staged copies under `paper/tables/`.

## Narrow B2 spot reruns

| Run | Rows | Main quality | Max p99 | Use in manuscript |
| --- | ---: | --- | ---: | --- |
| S1 FAISS CPU, bge-small | 9 | nDCG@10 = 0.505506 | 27.663 ms | Sanity check for strict-latency S1 result |
| S3 FAISS CPU, bge-small | 9 | evidence_coverage@10 = 0.851500 | 20.638 ms | Sanity check for strict-latency S3 result |
| S3 FAISS CPU, bge-base | 9 | evidence_coverage@10 = 0.871200 | 72.018 ms | Sanity check for S3 quality-tier behavior |

The spot reruns are deliberately not merged into the main paper tables. The main manuscript continues to cite `results/20260429T033427Z` as the paper bundle.

## S3 matched-query quality audit

The original S3 quality-first margin was too small to support a substantive pgvector quality claim, so we ran a direct matched-query audit over all 5,000 `HotpotQA-portable` S3 queries with bge-base, `clients_read=4`, and one observation per query.

| Setting | Queries | evidence_coverage@10 | p99 | QPS | Paired delta vs FAISS |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAISS CPU, HNSW32 | 5,000 | 0.8712 | 17.646 ms | 247.596 | reference |
| pgvector, IVF32 | 5,000 | 0.8711 | 306.044 ms | 15.214 | -0.0001, CI [-0.0003, 0.0000] |
| pgvector, IVF64 | 5,000 | 0.8711 | 346.169 ms | 14.584 | -0.0001, CI [-0.0003, 0.0000] |

Conclusion: the archived pgvector S3 quality-first row should be treated as objective sensitivity, not as a meaningful quality win. In the matched-query audit, pgvector is essentially tied or slightly lower in evidence coverage and much slower at p99.

## S2 strict-competitor quick check

A broad S2 rerun was started to compare FAISS CPU and Qdrant under the streaming-memory workload. The run was stopped because the full competitor matrix was too expensive for a quick manuscript check, but one complete FAISS observation file was available:

| Setting | Quality obs. | Freshness obs. | nDCG@10 | freshness_hit@5s | p99 | Errors | Use in manuscript |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FAISS CPU, HNSW32, bge-small | 894 | 500 | 0.505506 | 0.830 | 15.129 ms | 0 | Observation-level spot check only |

A narrower Qdrant HNSW64 run was then attempted with `paper/experiments/s2_single_sweep/qdrant_hnsw64.yaml`. It was interrupted at 42:11 with no result or observation files under `artifacts/runs/neurips_rerun/s2_qdrant_hnsw64_single/qdrant`. The traceback showed the run was still inside the S2 freshness probe sleep. This attempt should not be used as evidence; it indicates that S2 competitor reruns are not quick enough for opportunistic manuscript checks.

We then ran a dedicated longer standard B2 Qdrant HNSW64 row with `paper/experiments/s2_standard_competitor/qdrant_hnsw64.yaml`. It completed with `results.parquet` and a 1,394-line observation log:

| Setting | Quality obs. | Freshness obs. | nDCG@10 | freshness_hit@5s | p99 | Errors | Use in manuscript |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qdrant, HNSW64, bge-small | 894 | 500 | 0.502726 | 0.830 | 44.534 ms | 0 | Standard S2 competitor spot check |

Compared with the complete FAISS HNSW32 observation-level spot check, Qdrant matches freshness but is 0.0028 lower in nDCG@10 and has higher observed p99. This supports the S2 strict-decision story while keeping the claim local to the run.

Paired analysis over matched observation IDs strengthens this check:

- Static quality: Qdrant minus FAISS nDCG@10 = -0.002780, paired bootstrap 95% CI [-0.007268, 0.000574] over 894 matched queries.
- Freshness: Qdrant minus FAISS freshness_hit@5s = 0.000, paired bootstrap 95% CI [0.000, 0.000] over 500 matched events.
- Interpretation: the S2 competitor check does not prove a statistically decisive quality gap, but it does rule out a hidden Qdrant quality win in this setting while preserving equal freshness.

We then ran a larger same-machine FAISS/Qdrant S2 matrix with two repeats per engine, `phase_max_requests_per_phase=1000`, and `s2_max_freshness_events=100`. It completed with strict-schema validation over both run directories and 0 errors:

| Setting | Rows | Quality obs. | Freshness obs. | nDCG@10 mean | freshness_hit@5s | Mean / max p99 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS CPU, ef64, bge-small | 2 | 1,788 | 200 | 0.505506 | 0.840 | 47.763 / 74.148 ms | 0 |
| Qdrant, ef64, bge-small | 2 | 1,788 | 200 | 0.502835 | 0.840 | 17.078 / 20.714 ms | 0 |

Paired deltas are Qdrant minus FAISS:

- Static quality: nDCG@10 = -0.002671, paired bootstrap 95% CI [-0.005212, -0.000575] over 1,788 matched queries.
- Freshness: freshness_hit@5s = 0.000, paired bootstrap 95% CI [0.000, 0.000] over 200 matched events.
- Latency: query latency = -6.849 ms, paired bootstrap 95% CI [-8.944, -5.088] over 1,788 matched quality observations.

Interpretation: the larger same-machine run rules out a hidden Qdrant quality or freshness win in S2. Its latency direction differs from the earlier one-row Qdrant check, so manuscript wording should keep latency scoped to the specific local run and should not present this as a universal engine ranking.

## Attempted non-FAISS rerun

An S3 LanceDB-inproc bge-small B2 rerun was started to check the no-p99 cost winner. It was stopped after roughly 14 minutes because it had produced only partial observation files and no completed `results.parquet`, so it no longer qualified as a quick experiment. This reinforces that the no-p99 LanceDB result should be treated as an archived objective-sensitivity finding, not as a separately confirmed quick-rerun result.

## S3 quality-first margin check

The archived S3 bge-base rows do not contain per-query observation paths, so the first cheap pass used an aggregate-row bootstrap on all B2 rows. That gave pgvector-base minus FAISS-base evidence_coverage@10 = 0.00016 with a 95% bootstrap interval of -0.00280 to 0.00304 and bootstrap probability of a positive difference 0.543. The later matched-query audit above is stronger evidence and shows no substantive pgvector quality advantage.
