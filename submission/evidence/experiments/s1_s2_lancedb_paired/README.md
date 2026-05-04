# S1/S2 FAISS vs LanceDB Paired Audit

Bounded same-machine B2 audit for Task 4 in `improvements.md`.

Scope:

- Engines: `faiss-cpu` and `lancedb-inproc`.
- Embedding: `BAAI/bge-small-en-v1.5`.
- Workloads: S1 single-hop and S2 streaming memory.
- Matching: paired by repeat and query ID for quality; paired by repeat and event index for S2 post-insert events.
- S2 cap: 20 post-insert events per repeat to keep the paired check bounded.
- Purpose: check whether the strict FAISS choices over LanceDB are relevance wins or tail-latency tie-breaks.

Commands:

```bash
python -m maxionbench.cli run-matrix \
  --scenario-config-dir paper/experiments/s1_s2_lancedb_paired/scenarios \
  --engine-config-dir paper/experiments/s1_s2_lancedb_paired/engines \
  --out-dir artifacts/run_matrix/neurips_s1_s2_lancedb_paired \
  --output-root artifacts/runs/neurips_rerun/s1_s2_lancedb_paired_b2 \
  --budget b2 --lane cpu --json

python -m maxionbench.cli execute-run-matrix \
  --matrix artifacts/run_matrix/neurips_s1_s2_lancedb_paired/run_matrix.json \
  --lane cpu --budget b2 --no-retry --skip-completed \
  --engine-filter faiss-cpu,lancedb-inproc \
  --scenario-filter s1_single_hop,s2_streaming_memory \
  --template-filter s1_single_hop__bge-small-en-v1-5,s2_streaming_memory__bge-small-en-v1-5 \
  --deadline-hours 1 --json

```

The in-process LanceDB row requires the optional `lancedb` Python package.
