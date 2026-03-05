MaxionBench is an open, reproducible benchmark suite for retrieval backends and RAG pipeline components used in agentic LLM systems.

Implemented v0.1 harness components include:
- adapter contract + conformance suite
- scenarios `calibrate_d3`, `s1`..`s6`
- adapters: `mock`, `qdrant`, `pgvector`, `milvus`, `weaviate`, `opensearch`, `lancedb-service`, `lancedb-inproc`, `faiss-cpu`, `faiss-gpu`
- D4 local bundle loader for pinned BEIR subsets + CRAG slice
- report bundle generation for milestone/final figures and core paper tables (T1-T4)
- explicit RHU resource-profile capture (`resource_*`, `rhu_rate`) in `results.parquet` and run metadata (`rhu_references`, `resource_profile`) surfaced in T1 exports
- phased warmup/steady-state execution controls (`phase_timing_mode`, phase request caps)
- strict output validation plus legacy stage-timing backfill tooling (`maxionbench validate`, `maxionbench migrate-stage-timing`)

Artifact preflight before report generation:
1. Validate artifacts: `maxionbench validate --input artifacts/runs --json`
2. If validation reports missing stage timing columns, backfill legacy runs:
   - dry run: `maxionbench migrate-stage-timing --input artifacts/runs --dry-run`
   - apply: `maxionbench migrate-stage-timing --input artifacts/runs`
3. Re-validate: `maxionbench validate --input artifacts/runs --json`
4. Generate report: `maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones`

Pre-merge automation:
- `.github/workflows/report_preflight.yml` runs a fast smoke benchmark, validates artifacts with `maxionbench validate`, then runs `maxionbench report`.
- The workflow uploads smoke run/report artifacts (`actions/upload-artifact`) for debugging on both success and failure.
- The same workflow also exercises the legacy migration path: report failure on missing stage timing columns, migration backfill, re-validation, then successful report generation.
- The workflow enables pip dependency caching via `actions/setup-python` (`cache: pip`) to keep pre-merge runtime stable.
- Optional policy drift check: `maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json`
- To also require drift workflow status in verification: add `--include-drift-check`.
- Automated drift checker workflow: `.github/workflows/branch_protection_drift.yml`

Migration details:
- `docs/migrations/result_schema_stage_timing_v0_1.md`
- `docs/migrations/result_schema_resource_profile_v0_1.md`
- `docs/ci/branch_protection.md`
