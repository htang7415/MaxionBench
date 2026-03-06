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
- dataset manifest coverage + checksum format validation (`maxionbench verify-dataset-manifests`)
- D3 calibration paper-readiness verification (`maxionbench verify-d3-calibration`)
- behavior-card coverage validation (`maxionbench verify-behavior-cards`)
- conformance config catalog validation (`maxionbench verify-conformance-configs`)
- engine readiness gate from conformance matrix + behavior cards (`maxionbench verify-engine-readiness`)
- pre-run benchmark gate to block real-engine runs when readiness fails (`maxionbench pre-run-gate`)
- promotion gate from strict readiness summary artifacts (`maxionbench verify-promotion-gate`)
- S5 reranker runtime supports `hf_cross_encoder` when `MAXIONBENCH_ENABLE_HF_RERANKER=1` and local model/runtime deps are available, with explicit `heuristic_proxy` fallback provenance in `search_params_json` (pinned S5 configs set `s5_require_hf_backend: true` to fail fast if fallback occurs)

Artifact preflight before report generation:
1. Validate artifacts: `maxionbench validate --input artifacts/runs --strict-schema --json`
   - optional pinned protocol audit for paper-grade runs: `maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json`
   - `--enforce-protocol` also validates per-row robustness payloads:
   - S2 requires `search_params_json` keys `selectivity`, `filter`, `p99_inflation_vs_unfiltered`, and an explicit 100% anchor row (`selectivity=1.0`) with inflation `1.0`
  - S3/S3b require `search_params_json` keys `s1_baseline_p99_ms`, `s1_baseline_match_rows`, `s1_baseline_lookup_root`, `s1_baseline_missing`, `p99_inflation_vs_s1_baseline`, `burst_clock_anchor`; enforce `burst_clock_anchor="measurement_start"` plus consistent missing/non-missing baseline semantics
  - S3b additionally requires burst metadata keys `burst_on_s`, `burst_off_s`, `burst_cycle_s`, `burst_on_write_mult`, `burst_off_write_mult` and enforces `mode="s3_bursty"` with `burst_cycle_s = burst_on_s + burst_off_s`
   - S5 requires `search_params_json.reranker.backend="hf_cross_encoder"`, `device="cuda"`, `local_files_only=true`, `uses_qrels_supervision=false`, pinned reranker config fields, and `runtime_errors=0` with empty `fallback_reason`
   - run metadata must pin `rtt_baseline_request_profile="healthcheck_plus_query_topk1_zero_vector"` for cross-engine baseline comparability
     - run metadata must include `dataset_cache_checksums` provenance entries (`path_key`, `resolved_path`, `source`, `expected_sha256`, `actual_sha256`) when dataset checksum pins are provided
2. Verify D3 calibration params before D3 robustness runs: `maxionbench verify-d3-calibration --d3-params artifacts/calibration/d3_params.yaml --strict --json`
   - paper-ready calibration requires non-trivial eval metrics and calibration on real large-scale vectors (default minimum `10,000,000`)
3. If validation reports missing stage timing columns, backfill legacy runs:
   - dry run: `maxionbench migrate-stage-timing --input artifacts/runs --dry-run`
   - apply: `maxionbench migrate-stage-timing --input artifacts/runs`
4. Re-validate: `maxionbench validate --input artifacts/runs --strict-schema --json`
5. Generate report: `maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones/Mx`
   - T3 robustness export (`T3_robustness_summary.csv`) includes inflation-computability fields:
     - `p99_inflation_valid_rows`
     - `p99_inflation_nan_rows`
     - `p99_inflation_status` in `{computed_all_rows, computed_partial_rows, not_computable}`

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
- The workflow runs `conformance_readiness_gate` to generate `artifacts/conformance/conformance_matrix.csv` and enforce structural readiness coverage via `maxionbench verify-engine-readiness --allow-nonpass-status --require-mock-pass` (strict pass/fail readiness is handled in `strict_readiness.yml`).
- Optional strict readiness workflow for provisioned environments: `.github/workflows/strict_readiness.yml` (runs readiness without `--allow-nonpass-status` and installs `.[dev,engines]`).
- Optional publish workflow with strict-readiness artifact gate: `.github/workflows/publish_benchmark_bundle.yml`.
- The workflow runs `maxionbench verify-pins --config-dir configs/scenarios --json` before smoke generation.
- The workflow runs `maxionbench verify-behavior-cards --behavior-dir docs/behavior --json` to enforce behavior-card coverage/sections.
- The workflow runs `maxionbench verify-conformance-configs --config-dir configs/conformance --json` to enforce conformance config catalog shape and adapter coverage.
- The smoke path runs `maxionbench pre-run-gate --config ci_s1_smoke.yaml --json` before `maxionbench run`.
- For S5 configs with `s5_require_hf_backend: true`, pre-run gate also checks `MAXIONBENCH_ENABLE_HF_RERANKER`, local `torch`/`transformers` availability, and at least one visible NVIDIA GPU.
- The workflow uploads smoke run/report artifacts (`actions/upload-artifact`) for debugging on both success and failure.
- The same workflow also exercises the legacy migration path: report failure on missing stage timing columns, migration backfill, re-validation, then successful report generation.
- It also exercises a legacy resource-profile path: strict validation/report failure for missing RHU fields plus `--legacy-ok` warning diagnostics artifact.
- It also exercises a legacy ground-truth metadata path: strict validation/report failure for missing `ground_truth_*` fields plus `--legacy-ok` warning diagnostics artifact.
- The workflow also runs repository hygiene checks to guard against tracked Python cache artifacts.
- The workflow also runs command-doc consistency checks to prevent stale CLI examples.
- The workflow also runs figure-policy sync checks to keep report figure IDs and style pins aligned with `prompt.md`.
- The workflow also checks README migration index consistency against `docs/migrations/`.
- The workflow also checks branch-protection policy sync against report-preflight job names.
- The workflow also verifies Slurm plan consistency with `maxionbench verify-slurm-plan --json`.
- The workflow also verifies Slurm plan consistency for GPU-omitted mode with `maxionbench verify-slurm-plan --skip-gpu --json`.
- The workflow also verifies Slurm dependency planning with `maxionbench submit-slurm-plan --dry-run --json`.
- The workflow also verifies Slurm dependency planning for GPU-omitted mode with `maxionbench submit-slurm-plan --skip-gpu --dry-run --json`.
- The workflow also captures Slurm plan diagnostics under `artifacts/ci/` (`slurm_plan_verify*.json`, `slurm_submit_plan*_dry_run.json`).
- The workflow validates these Slurm diagnostic snapshots with `maxionbench validate-slurm-snapshots --json` before proceeding.
- The workflow writes `artifacts/ci/slurm_snapshot_validation.json` as the consolidated Slurm snapshot validation result.
- The workflow also writes a required-check snapshot artifact via `maxionbench snapshot-required-checks --strict`.
- The workflow also enforces report output-policy sidecar checks via `maxionbench inspect-report-output-policy --strict`.
- The workflow also runs `maxionbench ci-protocol-audit --strict` as a consolidated policy gate across scenario pins, dataset-manifest integrity, Slurm diagnostics, and report output-policy checks.
- The workflow writes `artifacts/ci/report_output_policy_summary.json` for output-policy inspection diagnostics.
- The workflow writes `artifacts/ci/ci_protocol_audit.json` as the consolidated CI policy audit artifact.
- The workflow enables pip dependency caching via `actions/setup-python` (`cache: pip`) to keep pre-merge runtime stable.
- Optional policy drift check: `maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json`
- To also require drift workflow status in verification: add `--include-drift-check`.
- To also require strict-readiness workflow status in verification: add `--include-strict-readiness-check`.
- To also require publish-benchmark-bundle workflow status in verification: add `--include-publish-bundle-check`.
- Automated drift checker workflow: `.github/workflows/branch_protection_drift.yml`
- Branch protection doc includes a policy-sync guard section describing the tests that enforce workflow/check-name alignment.

Migration details:
- `docs/migrations/result_schema_stage_timing_v0_1.md`
- `docs/migrations/result_schema_resource_profile_v0_1.md`
- `docs/migrations/result_schema_ground_truth_metadata_v0_1.md`
- `docs/migrations/result_schema_hardware_runtime_v0_1.md`
- `docs/ci/branch_protection.md`
