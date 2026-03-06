from __future__ import annotations

import re
from pathlib import Path


def _assert_common_commands(text: str) -> None:
    assert "maxionbench verify-pins --config-dir configs/scenarios --json" in text
    assert "maxionbench verify-dataset-manifests --manifest-dir maxionbench/datasets/manifests --json" in text
    assert "maxionbench verify-d3-calibration --d3-params artifacts/calibration/d3_params.yaml --strict --json" in text
    assert "maxionbench verify-slurm-plan --json" in text
    assert "maxionbench verify-slurm-plan --skip-gpu --json" in text
    assert "maxionbench submit-slurm-plan --dry-run --json" in text
    assert "maxionbench submit-slurm-plan --skip-gpu --dry-run --json" in text
    assert "CPU_D3_BASELINE_JOB_ID=$(sbatch --parsable --dependency=afterok:${CALIB_JOB_ID} --array=1" in text
    assert "CPU_D3_WORKLOADS_JOB_ID=$(sbatch --parsable --dependency=afterok:${CALIB_JOB_ID}:${CPU_D3_BASELINE_JOB_ID} --array=2-4" in text
    assert "CPU_NON_D3_JOB_ID=$(sbatch --parsable --dependency=afterok:${CALIB_JOB_ID} --array=0,5-6" in text
    assert "maxionbench verify-behavior-cards --behavior-dir docs/behavior --json" in text
    assert "--enforce-readiness" in text
    assert "--conformance-matrix artifacts/conformance/conformance_matrix.csv" in text
    assert "--behavior-dir docs/behavior" in text
    assert "--allow-gpu-unavailable" in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --json"
    ) in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --allow-gpu-unavailable --json"
    ) in text
    assert (
        "maxionbench verify-engine-readiness --conformance-matrix artifacts/conformance/conformance_matrix.csv "
        "--behavior-dir docs/behavior --allow-gpu-unavailable --allow-nonpass-status --require-mock-pass --json"
    ) in text
    assert (
        "maxionbench pre-run-gate --config configs/scenarios/s1_ann_frontier_qdrant.yaml "
        "--conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior --json"
    ) in text
    assert (
        "maxionbench pre-run-gate --config configs/scenarios/s1_ann_frontier_qdrant.yaml "
        "--conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior "
        "--allow-gpu-unavailable --json"
    ) in text
    assert (
        "maxionbench verify-promotion-gate --strict-readiness-summary "
        "artifacts/conformance_strict/engine_readiness_summary.json --json"
    ) in text
    assert "if summary provenance is non-strict" in text
    assert "allow_nonpass_status=true" in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --json" in text
    assert "maxionbench validate --input artifacts/runs --strict-schema --enforce-protocol --json" in text
    assert "`--enforce-protocol` robustness payload checks:" in text
    assert "S2 (`s2_filtered_ann`) rows must include `search_params_json` keys:" in text
    assert "`p99_inflation_vs_unfiltered`" in text
    assert "S3/S3b rows must include `search_params_json` keys:" in text
    assert "`p99_inflation_vs_s1_baseline`" in text
    assert "S3/S3b rows must set `burst_clock_anchor` to `measurement_start`." in text
    assert "`s1_baseline_error`" in text
    assert "run metadata must set `rtt_baseline_request_profile` to `healthcheck_plus_query_topk1_zero_vector`." in text
    assert "run metadata must include matching `dataset_cache_checksums` entries" in text
    assert "T3 robustness signaling columns (`T3_robustness_summary.csv`):" in text
    assert "`p99_inflation_valid_rows`" in text
    assert "`p99_inflation_nan_rows`" in text
    assert "`p99_inflation_status` in `{computed_all_rows, computed_partial_rows, not_computable}`" in text
    assert "maxionbench validate --input artifacts/runs --legacy-ok --json" in text
    assert "maxionbench migrate-stage-timing --input artifacts/runs --dry-run" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones/Mx" in text
    assert "maxionbench report --input artifacts/runs --mode milestones --milestone-id M3" in text
    assert "maxionbench snapshot-required-checks --output artifacts/ci/required_checks_snapshot.json --strict --json" in text
    assert "maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 --strict --json" in text
    assert (
        "maxionbench inspect-report-output-policy --input artifacts/figures/milestones/M3 "
        "--output artifacts/ci/report_output_policy_summary.json --strict --json"
    ) in text
    assert "maxionbench ci-protocol-audit" in text
    assert "--manifest-dir maxionbench/datasets/manifests" in text
    assert "--output artifacts/ci/ci_protocol_audit.json" in text
    assert "--require-report-policy" in text
    assert "exits with code `2` when mismatches are detected" in text
    assert "exits with code `2` when sidecar policy checks fail" in text
    assert "exits with code `2` when consolidated checks fail" in text
    assert "maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json" in text
    assert "--include-drift-check --json" in text
    assert "--include-strict-readiness-check --json" in text
    assert "--include-publish-bundle-check --json" in text
    assert "--include-strict-readiness-check --include-publish-bundle-check --json" in text
    assert "Preflight CI writes both policy artifacts together:" in text
    assert "`artifacts/ci/required_checks_snapshot.json`" in text
    assert "`artifacts/ci/report_output_policy_summary.json`" in text
    assert "`artifacts/ci/ci_protocol_audit.json`" in text
    assert "gh workflow run publish_benchmark_bundle.yml" in text
    assert "-f bundle_name=benchmark-result-bundle" in text


def _assert_no_stale_validate_invocation(text: str) -> None:
    stale_pattern = re.compile(r"maxionbench validate --input artifacts/runs --json")
    assert stale_pattern.search(text) is None


def test_command_docs_use_strict_validate_and_legacy_mode() -> None:
    command_md = Path("command.md")
    command_mac_md = Path("command-mac.md")
    assert command_md.exists()
    assert command_mac_md.exists()

    text_command = command_md.read_text(encoding="utf-8")
    text_mac = command_mac_md.read_text(encoding="utf-8")

    _assert_common_commands(text_command)
    _assert_common_commands(text_mac)
    _assert_no_stale_validate_invocation(text_command)
    _assert_no_stale_validate_invocation(text_mac)
