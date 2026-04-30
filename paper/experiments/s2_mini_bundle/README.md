# S2 deadline mini-bundle

This folder records a same-machine S2 rerun designed for deadline-time repeatability evidence. It is not a replacement for the standard S2 latency rows because the config caps static query phases and freshness probes.

Protocol:

- Scenario: `s2_streaming_memory`
- Budget: `b2`
- Engines: `faiss-cpu`, `qdrant`
- Embedding: `BAAI/bge-small-en-v1.5`
- Repeats: 2 per engine
- Search sweep: `hnsw_ef=64`
- Static cap: `phase_max_requests_per_phase=250`
- Freshness cap: `s2_max_freshness_events=40`

Artifacts:

- Matrix: `artifacts/run_matrix/neurips_s2_mini_bundle_b2_deadline/run_matrix.json`
- Runs: `artifacts/runs/neurips_rerun/s2_mini_bundle_b2_deadline`
- Summary: `s2_b2_deadline_mini_bundle_summary.json`

Main result:

- Both engines completed two rows with zero errors under the same orchestration.
- Qdrant minus FAISS paired nDCG@10 was `-0.0007` with 95% CI `[-0.0026, 0.0005]` over 500 matched quality observations.
- Freshness deltas at 1s and 5s were exactly `0.0000` over 80 matched freshness events.
