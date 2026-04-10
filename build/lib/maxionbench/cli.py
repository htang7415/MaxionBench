"""Top-level CLI for MaxionBench."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re

from maxionbench.conformance.run import main as conformance_main

_MILESTONE_ID_RE = re.compile(r"^M[0-9]+$")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="maxionbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark scenario")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--repeats", type=int, default=None)
    run_parser.add_argument("--no-retry", action="store_true")
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--d3-params", default=None)
    run_parser.add_argument("--enforce-readiness", action="store_true")
    run_parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    run_parser.add_argument("--behavior-dir", default="docs/behavior")
    run_parser.add_argument("--allow-gpu-unavailable", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate output artifacts")
    validate_parser.add_argument("--input", required=True)
    validate_mode_group = validate_parser.add_mutually_exclusive_group()
    validate_mode_group.add_argument("--strict-schema", action="store_true")
    validate_mode_group.add_argument("--legacy-ok", action="store_true")
    validate_parser.add_argument("--enforce-protocol", action="store_true")
    validate_parser.add_argument("--json", action="store_true")

    migrate_parser = subparsers.add_parser(
        "migrate-stage-timing",
        help="Backfill legacy results with stage timing columns",
    )
    migrate_parser.add_argument("--input", required=True)
    migrate_parser.add_argument("--dry-run", action="store_true")

    verify_branch_parser = subparsers.add_parser(
        "verify-branch-protection",
        help="Verify GitHub branch protection required checks",
    )
    verify_branch_parser.add_argument("--repo", required=True)
    verify_branch_parser.add_argument("--branch", default="main")
    verify_branch_parser.add_argument("--token", default=None)
    verify_branch_parser.add_argument("--timeout-s", type=float, default=10.0)
    verify_branch_parser.add_argument("--required-check", action="append", dest="required_checks", default=None)
    verify_branch_parser.add_argument("--include-drift-check", action="store_true")
    verify_branch_parser.add_argument("--include-strict-readiness-check", action="store_true")
    verify_branch_parser.add_argument("--include-publish-bundle-check", action="store_true")
    verify_branch_parser.add_argument("--json", action="store_true")

    verify_pins_parser = subparsers.add_parser(
        "verify-pins",
        help="Verify pinned scenario config values",
    )
    verify_pins_parser.add_argument("--config-dir", default="configs/scenarios")
    verify_pins_parser.add_argument("--allow-dev-calibrate-d3-scale", action="store_true")
    verify_pins_parser.add_argument("--strict-d3-scenario-scale", action="store_true")
    verify_pins_parser.add_argument("--json", action="store_true")

    verify_dataset_manifests_parser = subparsers.add_parser(
        "verify-dataset-manifests",
        help="Verify dataset manifest coverage and pinned metadata fields",
    )
    verify_dataset_manifests_parser.add_argument("--manifest-dir", default="maxionbench/datasets/manifests")
    verify_dataset_manifests_parser.add_argument("--json", action="store_true")

    download_d1_parser = subparsers.add_parser(
        "download-d1",
        help="Download a D1 ann-benchmarks HDF5 bundle into the local data cache",
    )
    download_d1_parser.add_argument("--dataset-name", required=True)
    download_d1_parser.add_argument("--output", default=None)
    download_d1_parser.add_argument("--force", action="store_true")
    download_d1_parser.add_argument("--timeout-s", type=float, default=60.0)
    download_d1_parser.add_argument("--json", action="store_true")

    download_datasets_parser = subparsers.add_parser(
        "download-datasets",
        help="Download the requested local/community dataset tree under dataset/",
    )
    download_datasets_parser.add_argument("--root", default="dataset")
    download_datasets_parser.add_argument("--cache-dir", default=".cache")
    download_datasets_parser.add_argument("--crag-examples", type=int, default=500)
    download_datasets_parser.add_argument("--skip-d1d2", action="store_true")
    download_datasets_parser.add_argument("--skip-d3", action="store_true")
    download_datasets_parser.add_argument("--skip-d4", action="store_true")
    download_datasets_parser.add_argument("--force", action="store_true")
    download_datasets_parser.add_argument("--timeout-s", type=float, default=60.0)
    download_datasets_parser.add_argument("--json", action="store_true")

    preprocess_datasets_parser = subparsers.add_parser(
        "preprocess-datasets",
        help="Normalize raw datasets into the canonical processed layout",
    )
    preprocess_datasets_parser.add_argument("mode", choices=["ann-hdf5", "d3-yfcc-raw", "d3-yfcc", "d3-explicit", "beir", "crag"])
    preprocess_datasets_parser.add_argument("--input", default=None)
    preprocess_datasets_parser.add_argument("--out", required=True)
    preprocess_datasets_parser.add_argument("--family", default=None)
    preprocess_datasets_parser.add_argument("--name", default=None)
    preprocess_datasets_parser.add_argument("--metric", default=None)
    preprocess_datasets_parser.add_argument("--base", default=None)
    preprocess_datasets_parser.add_argument("--queries", default=None)
    preprocess_datasets_parser.add_argument("--gt", default=None)
    preprocess_datasets_parser.add_argument("--filters", default=None)
    preprocess_datasets_parser.add_argument("--payloads", default=None)
    preprocess_datasets_parser.add_argument("--query-split", default=None)
    preprocess_datasets_parser.add_argument("--private-query-token", default=None)
    preprocess_datasets_parser.add_argument("--skip-payloads", action="store_true")
    preprocess_datasets_parser.add_argument("--split", default=None)
    preprocess_datasets_parser.add_argument("--max-examples", type=int, default=None)
    preprocess_datasets_parser.add_argument("--chunk-chars", type=int, default=None)
    preprocess_datasets_parser.add_argument("--overlap", type=int, default=None)
    preprocess_datasets_parser.add_argument("--json", action="store_true")

    verify_conformance_configs_parser = subparsers.add_parser(
        "verify-conformance-configs",
        help="Verify conformance config catalog shape and required adapter coverage",
    )
    verify_conformance_configs_parser.add_argument("--config-dir", default="configs/conformance")
    verify_conformance_configs_parser.add_argument("--allow-gpu-unavailable", action="store_true")
    verify_conformance_configs_parser.add_argument("--json", action="store_true")

    verify_d3_calibration_parser = subparsers.add_parser(
        "verify-d3-calibration",
        help="Verify D3 calibration params file is paper-ready",
    )
    verify_d3_calibration_parser.add_argument("--d3-params", default="artifacts/calibration/d3_params.yaml")
    verify_d3_calibration_parser.add_argument("--min-vectors", type=int, default=10_000_000)
    verify_d3_calibration_parser.add_argument("--strict", action="store_true")
    verify_d3_calibration_parser.add_argument("--json", action="store_true")

    verify_behavior_cards_parser = subparsers.add_parser(
        "verify-behavior-cards",
        help="Verify behavior-card coverage and required sections",
    )
    verify_behavior_cards_parser.add_argument("--behavior-dir", default="docs/behavior")
    verify_behavior_cards_parser.add_argument("--json", action="store_true")

    verify_engine_readiness_parser = subparsers.add_parser(
        "verify-engine-readiness",
        help="Verify conformance + behavior-card readiness",
    )
    verify_engine_readiness_parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    verify_engine_readiness_parser.add_argument("--behavior-dir", default="docs/behavior")
    verify_engine_readiness_parser.add_argument("--allow-gpu-unavailable", action="store_true")
    verify_engine_readiness_parser.add_argument("--allow-nonpass-status", action="store_true")
    verify_engine_readiness_parser.add_argument("--require-mock-pass", action="store_true")
    verify_engine_readiness_parser.add_argument("--target-adapter", default=None)
    verify_engine_readiness_parser.add_argument("--json", action="store_true")

    pre_run_gate_parser = subparsers.add_parser(
        "pre-run-gate",
        help="Run pre-run readiness gate for benchmark execution",
    )
    pre_run_gate_parser.add_argument("--config", required=True)
    pre_run_gate_parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    pre_run_gate_parser.add_argument("--behavior-dir", default="docs/behavior")
    pre_run_gate_parser.add_argument("--allow-gpu-unavailable", action="store_true")
    pre_run_gate_parser.add_argument("--json", action="store_true")

    verify_promotion_gate_parser = subparsers.add_parser(
        "verify-promotion-gate",
        help="Verify strict-readiness artifact before promotion",
    )
    verify_promotion_gate_parser.add_argument(
        "--strict-readiness-summary",
        default="artifacts/conformance_strict/engine_readiness_summary.json",
    )
    verify_promotion_gate_parser.add_argument("--conformance-matrix", default=None)
    verify_promotion_gate_parser.add_argument("--json", action="store_true")

    snapshot_checks_parser = subparsers.add_parser(
        "snapshot-required-checks",
        help="Write required-checks snapshot JSON artifact",
    )
    snapshot_checks_parser.add_argument("--output", default="artifacts/ci/required_checks_snapshot.json")
    snapshot_checks_parser.add_argument("--report-workflow", default=".github/workflows/report_preflight.yml")
    snapshot_checks_parser.add_argument("--drift-workflow", default=".github/workflows/branch_protection_drift.yml")
    snapshot_checks_parser.add_argument("--branch-protection-doc", default="docs/ci/branch_protection.md")
    snapshot_checks_parser.add_argument("--pr-template", default=".github/pull_request_template.md")
    snapshot_checks_parser.add_argument("--strict", action="store_true")
    snapshot_checks_parser.add_argument("--json", action="store_true")

    inspect_report_policy_parser = subparsers.add_parser(
        "inspect-report-output-policy",
        help="Inspect report metadata output-policy sidecars",
    )
    inspect_report_policy_parser.add_argument("--input", required=True)
    inspect_report_policy_parser.add_argument("--output", default=None)
    inspect_report_policy_parser.add_argument("--strict", action="store_true")
    inspect_report_policy_parser.add_argument("--json", action="store_true")

    report_parser = subparsers.add_parser("report", help="Generate milestone/final report artifacts")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--mode", required=True, choices=["milestones", "final"])
    report_parser.add_argument("--out", required=False)
    report_parser.add_argument("--milestone-id", default=None, help="Milestone ID (for example M3)")

    conformance_parser = subparsers.add_parser("conformance", help="Run adapter conformance tests")
    conformance_parser.add_argument("--adapter", default="mock")
    conformance_parser.add_argument("--adapter-options-json", default="{}")
    conformance_parser.add_argument("--collection", default="conformance")
    conformance_parser.add_argument("--dimension", type=int, default=4)
    conformance_parser.add_argument("--metric", default="ip")

    wait_adapter_parser = subparsers.add_parser(
        "wait-adapter",
        help="Poll an adapter healthcheck until it becomes ready",
    )
    wait_source = wait_adapter_parser.add_mutually_exclusive_group(required=True)
    wait_source.add_argument("--config", default=None)
    wait_source.add_argument("--adapter", default=None)
    wait_adapter_parser.add_argument("--adapter-options-json", default="{}")
    wait_adapter_parser.add_argument("--timeout-s", type=float, default=120.0)
    wait_adapter_parser.add_argument("--poll-interval-s", type=float, default=1.0)
    wait_adapter_parser.add_argument("--json", action="store_true")

    conformance_matrix_parser = subparsers.add_parser(
        "conformance-matrix",
        help="Run conformance tests for all adapter configs",
    )
    conformance_matrix_parser.add_argument("--config-dir", default="configs/conformance")
    conformance_matrix_parser.add_argument("--out-dir", default="artifacts/conformance")
    conformance_matrix_parser.add_argument("--timeout-s", type=float, default=300.0)
    conformance_matrix_parser.add_argument("--adapters", default="")

    args = parser.parse_args(argv)
    if args.command == "run":
        from maxionbench.orchestration.runner import main as run_main

        run_argv: list[str] = ["--config", args.config]
        if args.seed is not None:
            run_argv.extend(["--seed", str(args.seed)])
        if args.repeats is not None:
            run_argv.extend(["--repeats", str(args.repeats)])
        if args.no_retry:
            run_argv.append("--no-retry")
        if args.output_dir:
            run_argv.extend(["--output-dir", args.output_dir])
        if args.d3_params:
            run_argv.extend(["--d3-params", args.d3_params])
        if args.enforce_readiness:
            run_argv.append("--enforce-readiness")
        if args.conformance_matrix:
            run_argv.extend(["--conformance-matrix", args.conformance_matrix])
        if args.behavior_dir:
            run_argv.extend(["--behavior-dir", args.behavior_dir])
        if args.allow_gpu_unavailable:
            run_argv.append("--allow-gpu-unavailable")
        return run_main(run_argv)
    if args.command == "validate":
        from maxionbench.tools.validate_outputs import validate_path
        import json

        summary = validate_path(
            Path(args.input).resolve(),
            strict_schema=not bool(args.legacy_ok),
            enforce_protocol=bool(args.enforce_protocol),
        )
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "migrate-stage-timing":
        from maxionbench.tools.migrate_stage_timing import backfill_path
        import json

        summary = backfill_path(Path(args.input).resolve(), dry_run=args.dry_run)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-branch-protection":
        from maxionbench.tools.verify_branch_protection import main as verify_branch_main

        verify_argv: list[str] = ["--repo", args.repo, "--branch", args.branch, "--timeout-s", str(args.timeout_s)]
        if args.token:
            verify_argv.extend(["--token", args.token])
        for check in args.required_checks or []:
            verify_argv.extend(["--required-check", check])
        if args.include_drift_check:
            verify_argv.append("--include-drift-check")
        if args.include_strict_readiness_check:
            verify_argv.append("--include-strict-readiness-check")
        if args.include_publish_bundle_check:
            verify_argv.append("--include-publish-bundle-check")
        if args.json:
            verify_argv.append("--json")
        return verify_branch_main(verify_argv)
    if args.command == "verify-pins":
        from maxionbench.tools.verify_pins import main as verify_pins_main

        verify_argv: list[str] = ["--config-dir", args.config_dir]
        if args.allow_dev_calibrate_d3_scale:
            verify_argv.append("--allow-dev-calibrate-d3-scale")
        if args.strict_d3_scenario_scale:
            verify_argv.append("--strict-d3-scenario-scale")
        if args.json:
            verify_argv.append("--json")
        return verify_pins_main(verify_argv)
    if args.command == "verify-dataset-manifests":
        from maxionbench.tools.verify_dataset_manifests import main as verify_dataset_manifests_main

        verify_argv = ["--manifest-dir", args.manifest_dir]
        if args.json:
            verify_argv.append("--json")
        return verify_dataset_manifests_main(verify_argv)
    if args.command == "download-d1":
        from maxionbench.tools.download_d1 import main as download_d1_main

        download_argv = ["--dataset-name", args.dataset_name, "--timeout-s", str(args.timeout_s)]
        if args.output:
            download_argv.extend(["--output", args.output])
        if args.force:
            download_argv.append("--force")
        if args.json:
            download_argv.append("--json")
        return download_d1_main(download_argv)
    if args.command == "download-datasets":
        from maxionbench.tools.download_datasets import main as download_datasets_main

        download_argv: list[str] = [
            "--root",
            args.root,
            "--cache-dir",
            args.cache_dir,
            "--crag-examples",
            str(args.crag_examples),
            "--timeout-s",
            str(args.timeout_s),
        ]
        if args.skip_d1d2:
            download_argv.append("--skip-d1d2")
        if args.skip_d3:
            download_argv.append("--skip-d3")
        if args.skip_d4:
            download_argv.append("--skip-d4")
        if args.force:
            download_argv.append("--force")
        if args.json:
            download_argv.append("--json")
        return download_datasets_main(download_argv)
    if args.command == "preprocess-datasets":
        from maxionbench.tools.preprocess_datasets import main as preprocess_datasets_main

        required_by_mode = {
            "ann-hdf5": {
                "--input": args.input,
                "--family": args.family,
                "--name": args.name,
                "--metric": args.metric,
            },
            "d3-yfcc-raw": {
                "--input": args.input,
            },
            "d3-yfcc": {
                "--input": args.input,
            },
            "d3-explicit": {
                "--base": args.base,
                "--queries": args.queries,
                "--gt": args.gt,
                "--filters": args.filters,
            },
            "beir": {
                "--input": args.input,
                "--name": args.name,
            },
            "crag": {
                "--input": args.input,
            },
        }
        missing = [flag for flag, value in required_by_mode.get(args.mode, {}).items() if value in {None, ""}]
        if missing:
            parser.error(f"preprocess-datasets {args.mode} requires {' '.join(missing)}")

        preprocess_argv: list[str] = [args.mode, "--out", args.out]
        if args.input is not None:
            preprocess_argv.extend(["--input", args.input])
        if args.family is not None:
            preprocess_argv.extend(["--family", args.family])
        if args.name is not None:
            preprocess_argv.extend(["--name", args.name])
        if args.metric is not None:
            preprocess_argv.extend(["--metric", args.metric])
        if args.base is not None:
            preprocess_argv.extend(["--base", args.base])
        if args.queries is not None:
            preprocess_argv.extend(["--queries", args.queries])
        if args.gt is not None:
            preprocess_argv.extend(["--gt", args.gt])
        if args.filters is not None:
            preprocess_argv.extend(["--filters", args.filters])
        if args.payloads is not None:
            preprocess_argv.extend(["--payloads", args.payloads])
        if args.query_split is not None:
            preprocess_argv.extend(["--query-split", args.query_split])
        if args.private_query_token is not None:
            preprocess_argv.extend(["--private-query-token", args.private_query_token])
        if args.skip_payloads:
            preprocess_argv.append("--skip-payloads")
        if args.split is not None:
            preprocess_argv.extend(["--split", args.split])
        if args.max_examples is not None:
            preprocess_argv.extend(["--max-examples", str(args.max_examples)])
        if args.chunk_chars is not None:
            preprocess_argv.extend(["--chunk-chars", str(args.chunk_chars)])
        if args.overlap is not None:
            preprocess_argv.extend(["--overlap", str(args.overlap)])
        if args.json:
            preprocess_argv.append("--json")
        return preprocess_datasets_main(preprocess_argv)
    if args.command == "verify-conformance-configs":
        from maxionbench.tools.verify_conformance_configs import main as verify_conformance_configs_main

        verify_argv = ["--config-dir", args.config_dir]
        if args.allow_gpu_unavailable:
            verify_argv.append("--allow-gpu-unavailable")
        if args.json:
            verify_argv.append("--json")
        return verify_conformance_configs_main(verify_argv)
    if args.command == "verify-d3-calibration":
        from maxionbench.tools.verify_d3_calibration import main as verify_d3_calibration_main

        verify_argv = ["--d3-params", args.d3_params, "--min-vectors", str(args.min_vectors)]
        if args.strict:
            verify_argv.append("--strict")
        if args.json:
            verify_argv.append("--json")
        return verify_d3_calibration_main(verify_argv)
    if args.command == "verify-behavior-cards":
        from maxionbench.tools.verify_behavior_cards import main as verify_behavior_cards_main

        verify_argv: list[str] = ["--behavior-dir", args.behavior_dir]
        if args.json:
            verify_argv.append("--json")
        return verify_behavior_cards_main(verify_argv)
    if args.command == "verify-engine-readiness":
        from maxionbench.tools.verify_engine_readiness import main as verify_engine_readiness_main

        verify_argv: list[str] = [
            "--conformance-matrix",
            args.conformance_matrix,
            "--behavior-dir",
            args.behavior_dir,
        ]
        if args.allow_gpu_unavailable:
            verify_argv.append("--allow-gpu-unavailable")
        if args.allow_nonpass_status:
            verify_argv.append("--allow-nonpass-status")
        if args.require_mock_pass:
            verify_argv.append("--require-mock-pass")
        if args.target_adapter:
            verify_argv.extend(["--target-adapter", args.target_adapter])
        if args.json:
            verify_argv.append("--json")
        return verify_engine_readiness_main(verify_argv)
    if args.command == "pre-run-gate":
        from maxionbench.tools.pre_run_gate import main as pre_run_gate_main

        gate_argv: list[str] = [
            "--config",
            args.config,
            "--conformance-matrix",
            args.conformance_matrix,
            "--behavior-dir",
            args.behavior_dir,
        ]
        if args.allow_gpu_unavailable:
            gate_argv.append("--allow-gpu-unavailable")
        if args.json:
            gate_argv.append("--json")
        return pre_run_gate_main(gate_argv)
    if args.command == "verify-promotion-gate":
        from maxionbench.tools.verify_promotion_gate import main as verify_promotion_gate_main

        verify_argv: list[str] = ["--strict-readiness-summary", args.strict_readiness_summary]
        if args.conformance_matrix:
            verify_argv.extend(["--conformance-matrix", args.conformance_matrix])
        if args.json:
            verify_argv.append("--json")
        return verify_promotion_gate_main(verify_argv)
    if args.command == "snapshot-required-checks":
        from maxionbench.tools.required_checks_snapshot import main as snapshot_required_checks_main

        snapshot_argv: list[str] = [
            "--output",
            args.output,
            "--report-workflow",
            args.report_workflow,
            "--drift-workflow",
            args.drift_workflow,
            "--branch-protection-doc",
            args.branch_protection_doc,
            "--pr-template",
            args.pr_template,
        ]
        if args.strict:
            snapshot_argv.append("--strict")
        if args.json:
            snapshot_argv.append("--json")
        return snapshot_required_checks_main(snapshot_argv)
    if args.command == "inspect-report-output-policy":
        from maxionbench.tools.report_output_policy import main as inspect_report_output_policy_main

        inspect_argv: list[str] = ["--input", args.input]
        if args.output:
            inspect_argv.extend(["--output", args.output])
        if args.strict:
            inspect_argv.append("--strict")
        if args.json:
            inspect_argv.append("--json")
        return inspect_report_output_policy_main(inspect_argv)
    if args.command == "report":
        from maxionbench.reports.paper_exports import generate_report_bundle

        if args.mode == "final":
            if args.milestone_id:
                raise ValueError("--milestone-id is only valid when --mode milestones")
            if not args.out:
                raise ValueError("--out is required when --mode final")
            resolved_out = Path(args.out).resolve()
        else:
            if args.out and args.milestone_id:
                raise ValueError("Provide either --out or --milestone-id for milestone reports, not both")
            if args.milestone_id:
                if not _MILESTONE_ID_RE.fullmatch(str(args.milestone_id)):
                    raise ValueError("--milestone-id must match `M<integer>` (for example M2)")
                resolved_out = (Path("artifacts/figures/milestones") / str(args.milestone_id)).resolve()
            elif args.out:
                resolved_out = Path(args.out).resolve()
            else:
                raise ValueError("Milestone reports require either --out or --milestone-id")

        generate_report_bundle(
            input_dir=Path(args.input).resolve(),
            out_dir=resolved_out,
            mode=args.mode,
        )
        return 0
    if args.command == "conformance":
        conformance_argv = [
            "--adapter",
            args.adapter,
            "--adapter-options-json",
            args.adapter_options_json,
            "--collection",
            args.collection,
            "--dimension",
            str(args.dimension),
            "--metric",
            args.metric,
        ]
        return conformance_main(conformance_argv)
    if args.command == "wait-adapter":
        from maxionbench.tools.wait_adapter import main as wait_adapter_main

        wait_argv: list[str] = [
            "--timeout-s",
            str(args.timeout_s),
            "--poll-interval-s",
            str(args.poll_interval_s),
        ]
        if args.config:
            wait_argv.extend(["--config", args.config])
        if args.adapter:
            wait_argv.extend(["--adapter", args.adapter])
            wait_argv.extend(["--adapter-options-json", args.adapter_options_json])
        if args.json:
            wait_argv.append("--json")
        return wait_adapter_main(wait_argv)
    if args.command == "conformance-matrix":
        from maxionbench.conformance.matrix import main as conformance_matrix_main

        matrix_argv = [
            "--config-dir",
            args.config_dir,
            "--out-dir",
            args.out_dir,
            "--timeout-s",
            str(args.timeout_s),
        ]
        if str(args.adapters).strip():
            matrix_argv.extend(["--adapters", args.adapters])
        return conformance_matrix_main(matrix_argv)
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
