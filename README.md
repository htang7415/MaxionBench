MaxionBench is an open, reproducible benchmark suite for retrieval backends and RAG pipeline components used in agentic LLM systems.

Implemented v0.1 harness components include:
- adapter contract + conformance suite
- scenarios `calibrate_d3`, `s1`..`s6`
- adapters: `mock`, `qdrant`, `pgvector`, `milvus`, `weaviate`, `opensearch`, `lancedb-service`, `lancedb-inproc`, `faiss-cpu`, `faiss-gpu`
- D4 local bundle loader for pinned BEIR subsets + CRAG slice
- report bundle generation for milestone/final figures and core paper tables (T1-T4)
- explicit RHU resource-profile capture (`resource_*`, `rhu_rate`) in `results.parquet` and run metadata (`rhu_references`, `resource_profile`) surfaced in T1 exports
- explicit ground-truth provenance in run metadata (`ground_truth_source`, `ground_truth_metric`, `ground_truth_k`, `ground_truth_engine`) surfaced in T1 exports
- phased warmup/steady-state execution controls (`phase_timing_mode`, phase request caps)
- strict output validation plus legacy stage-timing backfill tooling (`maxionbench validate`, `maxionbench migrate-stage-timing`)
- pinned scenario config drift detection (`maxionbench verify-pins`)
- behavior-card coverage validation (`maxionbench verify-behavior-cards`)
- engine readiness gate from conformance matrix + behavior cards (`maxionbench verify-engine-readiness`)
- pre-run benchmark gate to block real-engine runs when readiness fails (`maxionbench pre-run-gate`)

Artifact preflight before report generation:
1. Validate artifacts: `maxionbench validate --input artifacts/runs --strict-schema --json`
   - optional pinned protocol audit for paper-grade runs: `maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json`
2. If validation reports missing stage timing columns, backfill legacy runs:
   - dry run: `maxionbench migrate-stage-timing --input artifacts/runs --dry-run`
   - apply: `maxionbench migrate-stage-timing --input artifacts/runs`
3. Re-validate: `maxionbench validate --input artifacts/runs --strict-schema --json`
4. Generate report: `maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones/Mx`

Report sidecar output-policy inspection:
1. Inspect sidecars: `maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 --strict --json`
   - optional artifact output: add `--output artifacts/ci/report_output_policy_summary.json`
2. Expected strict success:
   - `"pass": true`
   - `output_path_class_counts` has one expected class for the target directory (`milestones_mx`, `milestones_noncanonical`, or `final`)
3. If strict inspection fails, common remediation:
   - regenerate the report bundle with current tooling
   - ensure all `*.meta.json` files include `output_policy` keys (`mode`, `resolved_out_dir`, `output_path_class`, `milestone_id`)
   - rerun inspection with `--strict --json` and verify `error_count: 0`
4. Strict mode exit codes for policy audits:
   - `maxionbench snapshot-required-checks --strict ...` exits with code `2` on required-check drift
   - `maxionbench inspect-report-output-policy --strict ...` exits with code `2` on sidecar policy drift

Common `inspect-report-output-policy` failures and fixes:

| Summary key | Typical meaning | Fix action |
| --- | --- | --- |
| `missing_output_policy_files` | One or more sidecars exist but do not include an `output_policy` object | Regenerate report outputs with current code path (`maxionbench report ...`) and avoid manual sidecar edits |
| `invalid_output_policy` | `output_policy` exists but fields/values are malformed (for example wrong class/id combination) | Re-run report generation with the correct mode/output path (`--mode milestones` + `--milestone-id Mx` or a valid `--out`) |
| `invalid_json_files` | At least one `*.meta.json` file is not valid JSON | Replace/regen the corrupted sidecar by rerunning report generation for that output directory |
| `no_meta_files` | No `*.meta.json` sidecars were found in the inspected directory | Confirm `--input` points to a report output directory and regenerate report artifacts if needed |
| `mixed_output_path_classes` / `mixed_modes` / `mixed_milestone_ids` | Sidecars in one directory disagree on output-policy class/mode/milestone | Regenerate the full report directory in one run and avoid mixing copied sidecars from different runs |
| `output_path_class_mismatch` / `milestone_id_mismatch` | Sidecars are self-consistent but do not match what the inspected directory implies (for example `artifacts/figures/milestones/M3` must be `milestones_mx` + `M3`) | Regenerate the bundle in-place and avoid manually editing class/id fields |
| `resolved_out_dir_mismatch` / `milestone_root_mismatch` | Sidecars are internally consistent but reference a different output directory/root than `--input` | Regenerate sidecars for the current output directory and avoid copying `*.meta.json` files across bundles |

Legacy compatibility mode:
- Use `maxionbench validate --input artifacts/runs --legacy-ok --json` for local inspection of older artifacts.
- CI/report preflight should keep strict schema validation (default behavior without `--legacy-ok`).

Pre-merge automation:
- `.github/workflows/report_preflight.yml` runs a fast smoke benchmark, validates artifacts with `maxionbench validate --strict-schema`, then runs `maxionbench report`.
- The workflow runs `conformance_readiness_gate` to generate `artifacts/conformance/conformance_matrix.csv` and enforce readiness coverage via `maxionbench verify-engine-readiness --allow-nonpass-status`.
- Optional strict readiness workflow for provisioned environments: `.github/workflows/strict_readiness.yml` (runs readiness without `--allow-nonpass-status`).
- The workflow runs `maxionbench verify-pins --config-dir configs/scenarios --json` before smoke generation.
- The workflow runs `maxionbench verify-behavior-cards --behavior-dir docs/behavior --json` to enforce behavior-card coverage/sections.
- The smoke path runs `maxionbench pre-run-gate --config ci_s1_smoke.yaml --json` before `maxionbench run`.
- The workflow uploads smoke run/report artifacts (`actions/upload-artifact`) for debugging on both success and failure.
- The same workflow also exercises the legacy migration path: report failure on missing stage timing columns, migration backfill, re-validation, then successful report generation.
- It also exercises a legacy resource-profile path: strict validation/report failure for missing RHU fields plus `--legacy-ok` warning diagnostics artifact.
- It also exercises a legacy ground-truth metadata path: strict validation/report failure for missing `ground_truth_*` fields plus `--legacy-ok` warning diagnostics artifact.
- The workflow also runs repository hygiene checks to guard against tracked Python cache artifacts.
- The workflow also runs command-doc consistency checks to prevent stale CLI examples.
- The workflow also runs figure-policy sync checks to keep report figure IDs and style pins aligned with `prompt.md`.
- The workflow also checks README migration index consistency against `docs/migrations/`.
- The workflow also checks branch-protection policy sync against report-preflight job names.
- The workflow also writes a required-check snapshot artifact via `maxionbench snapshot-required-checks --strict`.
- The workflow also enforces report output-policy sidecar checks via `maxionbench inspect-report-output-policy --strict`.
- The workflow writes `artifacts/ci/report_output_policy_summary.json` for output-policy inspection diagnostics.
- The workflow enables pip dependency caching via `actions/setup-python` (`cache: pip`) to keep pre-merge runtime stable.
- Optional policy drift check: `maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json`
- To also require drift workflow status in verification: add `--include-drift-check`.
- Automated drift checker workflow: `.github/workflows/branch_protection_drift.yml`
- Branch protection doc includes a policy-sync guard section describing the tests that enforce workflow/check-name alignment.

Migration details:
- `docs/migrations/result_schema_stage_timing_v0_1.md`
- `docs/migrations/result_schema_resource_profile_v0_1.md`
- `docs/migrations/result_schema_ground_truth_metadata_v0_1.md`
- `docs/migrations/result_schema_hardware_runtime_v0_1.md`
- `docs/ci/branch_protection.md`
