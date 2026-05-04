# S3 all-evidence hit audit

This folder records the targeted query-level audits used to fill
`all_evidence_hit@10` for S3 rows whose archived B2 aggregate table did not
store per-query retrieval observations.

The audit uses the same `HOTPOTQA-MAXIONBENCH` 5,000-query cap recorded in the
paper run configuration. It should be read as query-level support for the
binary all-evidence metric, not as a replacement for the archived B2 deployment
matrix.

Run commands:

```bash
python -c "from pathlib import Path; from maxionbench.orchestration.runner import run_from_config; run_from_config(Path('paper/experiments/s3_all_evidence/lancedb_small.yaml'))"
python -c "from pathlib import Path; from maxionbench.orchestration.runner import run_from_config; run_from_config(Path('paper/experiments/s3_all_evidence/qdrant_small.yaml'))"
```
