# S2 larger same-machine comparison

This experiment is the deadline-safe S2 competitor check for the NeurIPS manuscript.

- Scenario: `s2_streaming_memory`
- Engines: `faiss-cpu`, `qdrant`
- Machine scope: same local machine as the other paper runs
- Budget: B2
- Embedding: `BAAI/bge-small-en-v1.5`
- Clients: 8 read, 2 write
- Search sweep: `hnsw_ef=64`
- Repeats: 2
- Static cap: `phase_max_requests_per_phase=1000`
- Post-insert cap: `s2_max_freshness_events=100`

This run is larger than the bounded replication bundle (`250` static requests and `40` post-insert events per repeat) while staying bounded enough to finish before submission. It should be interpreted as stronger same-orchestration repeatability evidence, not as a replacement for a fully uncapped S2 study.

## Result

- Matrix: `artifacts/run_matrix/neurips_s2_larger_same_machine_b2/run_matrix.json`
- Output root: `artifacts/runs/neurips_rerun/s2_larger_same_machine_b2`
- Summary: `paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json`
- Validation: `python -m maxionbench.cli validate --input artifacts/runs/neurips_rerun/s2_larger_same_machine_b2 --strict-schema --json` passed with 2 run directories and 0 errors.

Main result:

| Engine | Rows | Quality obs. | Post-insert obs. | nDCG@10 mean | post_insert_hit@10,5s | Mean / max p99 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS CPU | 2 | 1,788 | 200 | 0.505506 | 0.840 | 47.763 / 74.148 ms | 0 |
| Qdrant | 2 | 1,788 | 200 | 0.502835 | 0.840 | 17.078 / 20.714 ms | 0 |

Paired deltas are Qdrant minus FAISS:

- nDCG@10: -0.002671, 95% CI [-0.005212, -0.000575] over 1,788 matched quality observations.
- post_insert_hit@10,5s: 0.000, 95% CI [0.000, 0.000] over 200 matched post-insert events.
- latency_ms: -6.849 ms, 95% CI [-8.944, -5.088] over 1,788 matched quality observations.

Interpretation: this run rules out a hidden Qdrant quality or post-insert retrievability win in the S2 setting, but its latency direction differs from the earlier one-row standard Qdrant check. Use it as stronger local repeatability evidence and keep latency claims scoped to the measured run.
