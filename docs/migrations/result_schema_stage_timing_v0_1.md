# Result Schema Migration: Stage Timing Fields (v0.1)

Date: `2026-03-04`

## Change summary

`results.parquet` rows now require these timing fields:

- `setup_elapsed_s`
- `warmup_elapsed_s`
- `measure_elapsed_s`
- `export_elapsed_s`

Validation now fails if any of these fields are missing, non-numeric, or negative.

## Why this change exists

The benchmark report stage-timing figure (`m2_runner_stage_timing`) now uses explicit stage durations, so timing fields must be first-class schema columns instead of inferred placeholders.

## Backfill procedure for legacy runs

Dry-run first:

```bash
maxionbench migrate-stage-timing --input artifacts/runs --dry-run
```

Apply migration:

```bash
maxionbench migrate-stage-timing --input artifacts/runs
```

Then validate:

```bash
maxionbench validate --input artifacts/runs --json
```

## Notes

- Backfill sets missing stage timing fields to `0.0`.
- Existing timing values are preserved.
- Migration supports either a single run directory or a parent directory containing multiple runs.
