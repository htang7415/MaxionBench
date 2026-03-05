# Result Schema Migration: RHU Resource Profile Fields (v0.1)

Date: `2026-03-04`

## Change summary

`results.parquet` rows now require these RHU resource fields:

- `resource_cpu_vcpu`
- `resource_gpu_count`
- `resource_ram_gib`
- `resource_disk_tb`
- `rhu_rate`

`run_metadata.json` now requires:

- `rhu_references` with keys `c_ref_vcpu`, `g_ref_gpu`, `r_ref_gib`, `d_ref_tb`
- `resource_profile` with keys `cpu_vcpu`, `gpu_count`, `ram_gib`, `disk_tb`, `rhu_rate`

Validation now fails if any of these fields are missing, non-numeric, or negative.

## Why this change exists

RHU-hour reporting must be auditable. Explicit per-run resource profiles and pinned RHU reference values ensure T1 exports and downstream analyses can be reproduced exactly.

## Migration guidance for legacy runs

There is no lossless backfill for missing RHU resource profile fields in legacy outputs.

Recommended path:

1. Re-run benchmarks with current runner.
2. Validate artifacts:

```bash
maxionbench validate --input artifacts/runs --json
```

3. Generate reports:

```bash
maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones
```
