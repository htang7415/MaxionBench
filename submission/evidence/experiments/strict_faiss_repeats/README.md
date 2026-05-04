# Strict FAISS Same-Machine Repeat Audit

This audit summarizes completed same-machine repeat artifacts for the strict-latency FAISS CPU + `BAAI/bge-small-en-v1.5` choices. It is supplementary local-repeat evidence only; it does not replace the archived main decision tables and does not provide hardware-population latency intervals.

Inputs:

- S1: `artifacts/runs/neurips_rerun/strict_faiss_small/s1_single_hop__bge-small-en-v1-5/faiss_cpu/results.parquet`
- S2: `artifacts/runs/neurips_rerun/s2_larger_same_machine_b2/s2_streaming_memory__bge-small-en-v1-5/faiss_cpu/results.parquet`
- S3: `artifacts/runs/neurips_rerun/s3_faiss_small/s3_multi_hop__bge-small-en-v1-5/faiss_cpu/results.parquet`

Outputs:

- `paper/experiments/strict_faiss_repeats/strict_faiss_repeat_summary.json`
- `paper/tables/strict_faiss_repeat_audit.csv`
- `paper/tables/strict_faiss_repeat_audit.tex`
- `paper/manuscript/tables/strict_faiss_repeat_audit.tex`
