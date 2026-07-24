# Result Schema Migration: Hardware/Runtime Metadata (v0.1)

## Summary

`run_metadata.json` now requires a `hardware_runtime` mapping so each run records a minimal runtime/hardware summary for reproducibility.

Strict artifact validation (`maxionbench validate --strict-schema`) now fails if this mapping is missing or malformed.

## Required `hardware_runtime` keys

- `hostname`
- `platform`
- `python_version`
- `cpu_count_logical`
- `slurm_job_id`
- `slurm_array_task_id`
- `container_runtime_hint`
- `total_memory_bytes`
- `gpu_count`

Notes:
- string fields (`hostname`, `platform`, `python_version`) must be non-empty
- numeric fields (`cpu_count_logical`, `total_memory_bytes`, `gpu_count`) must be numeric and non-negative
- Slurm/container fields may be `null` when not applicable

## Impact

- Newly generated runs already include this mapping via `collect_system_info()`.
- Legacy runs missing `hardware_runtime` fail strict validation and report preflight.

## Remediation

Preferred:
- re-run benchmarks with current runner to regenerate artifacts

Temporary local inspection only:
- `maxionbench validate --input <runs_dir> --legacy-ok --json`

For report generation, strict mode remains required; legacy artifacts should be regenerated.
