# MaxionBench v0.1

MaxionBench is an open, reproducible benchmark suite for retrieval backends and RAG pipeline components used in agentic LLM systems.

Use `project.md`, `prompt.md`, `document.md`, and `command.md` in that order. This README is the quick-start and workflow index.

## Quick Start

Install the local environment:

```bash
pip install -e '.[dev,engines]'
```

Day-to-day development should use the non-paper configs under `configs/scenarios/`. Paper-grade runs use `configs/scenarios_paper/`.

Chosen dataset set for the current project direction:

- D1: `glove-100-angular`, `sift-128-euclidean`, `gist-960-euclidean`
- D2: `deep-image-96-angular`
- D3: `yfcc-10M`
- D4: BEIR `scifact`, `fiqa`, `nfcorpus` plus `crag_task_1_and_2_dev_v4.first_500.jsonl`

Bootstrap the documented local dataset tree with:

```bash
python -m maxionbench.cli download-datasets --root dataset --cache-dir .cache --crag-examples 500 --json
```

Recommended starting points:

```bash
maxionbench run --config configs/scenarios/s1_ann_frontier.yaml
maxionbench validate --input artifacts/runs --strict-schema --json
maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json
```

This is an optional pinned protocol audit for paper-grade runs. `maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json` is stricter than schema validation alone.

## Dev vs Paper

- Local development may use `phase_timing_mode: bounded` for smoke coverage and fast debugging.
- Paper-grade runs must use strict timing, verified dataset checksum pins, and verified D3 calibration artifacts.
- D3 robustness scenarios require a checked calibration artifact passed via `--d3-params`; the checked-in sample artifact is not paper-ready.
- `s5_require_hf_backend: true` is the paper default, and the reranker path expects `MAXIONBENCH_ENABLE_HF_RERANKER=1`, a visible NVIDIA GPU, and `device="cuda"`.
- The D3-matched S1 run is a robustness-accounting support baseline, not a headline S1 D1/D2 result.

## D3 Calibration

Run calibration before S2/S3/S3b so reported S2 runs stay benchmark results rather than tuning runs:

```bash
export MAXIONBENCH_D3_DATASET_PATH=/path/to/d3_vectors.npy
export MAXIONBENCH_D3_DATASET_SHA256=<sha256>
maxionbench verify-d3-calibration --d3-params artifacts/calibration/d3_params.yaml --strict --json
```

Paper and CI workflows use `require_paper_d3_calibration`, `d3_params_path`, and `strict_d3_scenario_scale`.

- `MAXIONBENCH_D3_DATASET_PATH`
- `MAXIONBENCH_D3_DATASET_SHA256`
- D3-50M runs reuse the frozen D3 calibration affinities from the 10M paper calibration

## Protocol Validation

`--enforce-protocol` also validates per-row robustness payloads:

- S2 requires `search_params_json` keys `selectivity`, `filter`, `p99_inflation_vs_unfiltered`
- S2 requires an explicit 100% anchor row (`selectivity=1.0`) with inflation `1.0`
- S3/S3b require `search_params_json` keys `s1_baseline_p99_ms`, `s1_baseline_match_rows`, `s1_baseline_lookup_root`, `s1_baseline_missing`, `p99_inflation_vs_s1_baseline`
- S3/S3b require `burst_clock_anchor`
- S3/S3b require `burst_clock_anchor="measurement_start"`
- S3b additionally requires burst metadata keys `burst_on_s`, `burst_off_s`, `burst_cycle_s`
- `mode="s3_bursty"`
- `burst_cycle_s = burst_on_s + burst_off_s`
- S5 requires `search_params_json.reranker.backend="hf_cross_encoder"`
- `device="cuda"`
- `local_files_only=true`
- `uses_qrels_supervision=false`
- `runtime_errors=0`
- empty `fallback_reason`
- `rtt_baseline_request_profile="healthcheck_plus_query_topk1_zero_vector"`
- `dataset_cache_checksums` provenance entries
- T3_robustness_summary.csv
- `p99_inflation_valid_rows`
- `p99_inflation_nan_rows`
- `p99_inflation_status` in `{computed_all_rows, computed_partial_rows, not_computable}`

## Readiness and CI

The readiness flow is built around:

- `verify-behavior-cards`
- `verify-engine-readiness`
- `verify-promotion-gate`
- `verify-dataset-manifests`
- `verify-d3-calibration`
- `pre-run-gate`
- `ci-protocol-audit --strict`

Useful commands:

```bash
maxionbench pre-run-gate --config ci_s1_smoke.yaml --json
maxionbench verify-conformance-configs --config-dir configs/conformance --json
maxionbench verify-slurm-plan --json
maxionbench verify-slurm-plan --skip-gpu --json
maxionbench validate-slurm-snapshots --json
maxionbench ci-protocol-audit --strict --output artifacts/ci/ci_protocol_audit.json
```

Notes for readiness and publish:

- `strict_readiness.yml`
- `publish_benchmark_bundle.yml`
- `conformance_readiness_gate`
- `include-strict-readiness-check`
- `include-publish-bundle-check`
- `scenario_config_dir`
- `allow-nonpass-status`
- non-pass rows fail readiness except `faiss-gpu` when `--allow-gpu-unavailable` is active
- non-pass rows are allowed only for `faiss-gpu`
- GPU-omitted mode (`--skip-gpu` / `allow_gpu_unavailable`) omits the GPU array entirely
- `verify-promotion-gate` cross-checks both strict summary and downloaded conformance matrix
- it requires a `mock` row with `status=pass` in the matrix artifact
- `slurm_snapshot_validation.json`
- `ci_protocol_audit.json`

## Slurm and Workstation

Use `run_workstation.sh` for preflight and Slurm submission planning. The Slurm profile system uses `profiles_local.yaml` and `profiles_local.example.yaml`.

Dataset and config notes:

- `prefetch_datasets.sh`
- `MAXIONBENCH_PREFETCH_D3_SOURCE`
- `MAXIONBENCH_PREFETCH_D4_BEIR_SOURCE`
- submit-slurm-plan uses the override file when present and otherwise falls back to its default scenario config
- the same override also applies to `calibrate_d3`
- `MAXIONBENCH_CALIBRATE_CONFIG` is explicitly set

Example commands:

```bash
maxionbench submit-slurm-plan --dry-run --json
maxionbench submit-slurm-plan --prefetch-datasets --dry-run --json
maxionbench submit-slurm-plan --skip-gpu --dry-run --json
maxionbench submit-slurm-plan --scenario-config-dir configs/scenarios_paper --skip-gpu --dry-run --json
maxionbench submit-slurm-plan --scenario-config-dir <dir> --container-runtime apptainer --container-image <path>
```

Reference string used in workflow docs: submit-slurm-plan --container-runtime apptainer --container-image <path>

Submission flags you will commonly need:

- `--container-bind`
- `--hf-cache-dir`

## Containers

The repo ships a `Dockerfile`. A common path is to build with Docker and run on Slurm with Apptainer.

```bash
docker build -t maxionbench:0.1.0 .
docker save maxionbench:0.1.0 -o maxionbench.tar
apptainer build ... docker-daemon://maxionbench:0.1.0
apptainer build maxionbench.sif docker-archive://...
```

## Publish and Promotion

The promotion gate verifies behavior cards, engine readiness, and publish bundle readiness before promotion:

- `verify-promotion-gate`
- `publish_benchmark_bundle.yml`
- `strict_readiness.yml`

This keeps paper-grade results aligned with `configs/scenarios_paper/`, strict timing, checksum pins, and paper-ready D3 calibration.

## Report Output Policy

Report sidecar output-policy inspection:

```bash
maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 --strict --json --output artifacts/ci/report_output_policy_summary.json
```

Expected summary snippets include `"pass": true` and `output_path_class_counts`. If the audit fails, rerun inspection with `--strict --json` and verify `error_count: 0`.

Strict mode exit codes for policy audits:

- `snapshot-required-checks --strict ...` exits with code `2`
- `inspect-report-output-policy --strict ...` exits with code `2`

The pre-merge workflow also documents:

- `verify-behavior-cards --behavior-dir docs/behavior --json`
- it enforces report output-policy sidecar checks
- `maxionbench inspect-report-output-policy --strict`
- `artifacts/ci/report_output_policy_summary.json`

Common `inspect-report-output-policy` failures and fixes:

| Summary key | Typical meaning | Fix action |
| --- | --- | --- |
| `missing_output_policy_files` | Sidecars were not emitted next to figures. | Re-render figures and ensure sidecars are written. |
| `invalid_output_policy` | Sidecar content is malformed or incomplete. | Regenerate the sidecar from the current report pipeline. |
| `invalid_json_files` | One or more sidecars are not valid JSON. | Rewrite the invalid JSON and rerun the inspection. |
| `no_meta_files` | Figures are missing required `*.meta.json` files. | Export the figures again with metadata enabled. |
| `mixed_output_path_classes` / `mixed_modes` / `mixed_milestone_ids` | A directory mixes incompatible figure classes or milestone tags. | Split outputs into consistent directories. |
| `output_path_class_mismatch` / `milestone_id_mismatch` | The sidecar metadata does not match the output path. | Correct the metadata or move the output into the matching path. |
| `resolved_out_dir_mismatch` / `milestone_root_mismatch` | The report resolved a different root than the artifact layout. | Align the report configuration and rerun export. |

Migration details:

- `docs/migrations/result_schema_ground_truth_metadata_v0_1.md`
- `docs/migrations/result_schema_hardware_runtime_v0_1.md`
- `docs/migrations/result_schema_resource_profile_v0_1.md`
- `docs/migrations/result_schema_stage_timing_v0_1.md`
- `docs/ci/branch_protection.md`
