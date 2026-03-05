# Result Schema Migration: Ground-Truth Metadata Fields (v0.1)

Date: `2026-03-04`

## Change summary

`run_metadata.json` strict validation now enforces:

- `ground_truth_source` (non-empty string)
- `ground_truth_metric` (non-empty string)
- `ground_truth_k` (integer, `>= 0`)
- `ground_truth_engine` (non-empty string)

In `--legacy-ok` mode, missing/invalid ground-truth fields are reported as warnings.

## Why this change exists

Ground-truth provenance must be explicit for reproducibility and auditability.
These fields identify where relevance labels or exact neighbors came from and how they were computed.

## Migration guidance for legacy runs

1. Re-run benchmarks with current runner to generate complete metadata.
2. Validate outputs strictly:

```bash
maxionbench validate --input artifacts/runs --strict-schema --json
```

3. For local inspection of older artifacts only:

```bash
maxionbench validate --input artifacts/runs --legacy-ok --json
```
