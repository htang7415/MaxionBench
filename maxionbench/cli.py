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
    verify_pins_parser.add_argument("--json", action="store_true")

    verify_dataset_manifests_parser = subparsers.add_parser(
        "verify-dataset-manifests",
        help="Verify dataset manifest coverage and pinned metadata fields",
    )
    verify_dataset_manifests_parser.add_argument("--manifest-dir", default="maxionbench/datasets/manifests")
    verify_dataset_manifests_parser.add_argument("--json", action="store_true")

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

    verify_slurm_plan_parser = subparsers.add_parser(
        "verify-slurm-plan",
        help="Verify Slurm submit-plan consistency with array scenario layout",
    )
    verify_slurm_plan_parser.add_argument("--slurm-dir", default="maxionbench/orchestration/slurm")
    verify_slurm_plan_parser.add_argument("--skip-gpu", action="store_true")
    verify_slurm_plan_parser.add_argument("--json", action="store_true")

    validate_slurm_snapshots_parser = subparsers.add_parser(
        "validate-slurm-snapshots",
        help="Validate Slurm plan snapshot JSON artifacts",
    )
    validate_slurm_snapshots_parser.add_argument("--verify-path", action="append", dest="verify_paths", default=None)
    validate_slurm_snapshots_parser.add_argument("--submit-path", action="append", dest="submit_paths", default=None)
    validate_slurm_snapshots_parser.add_argument(
        "--required-baseline-scenario",
        default="configs/scenarios/s1_ann_frontier_d3.yaml",
    )
    validate_slurm_snapshots_parser.add_argument("--json", action="store_true")

    ci_protocol_audit_parser = subparsers.add_parser(
        "ci-protocol-audit",
        help="Run consolidated CI protocol checks",
    )
    ci_protocol_audit_parser.add_argument("--config-dir", default="configs/scenarios")
    ci_protocol_audit_parser.add_argument("--slurm-dir", default="maxionbench/orchestration/slurm")
    ci_protocol_audit_parser.add_argument("--manifest-dir", default="maxionbench/datasets/manifests")
    ci_protocol_audit_parser.add_argument("--verify-path", action="append", dest="verify_paths", default=None)
    ci_protocol_audit_parser.add_argument("--submit-path", action="append", dest="submit_paths", default=None)
    ci_protocol_audit_parser.add_argument(
        "--required-baseline-scenario",
        default="configs/scenarios/s1_ann_frontier_d3.yaml",
    )
    ci_protocol_audit_parser.add_argument("--report-input", default=None)
    ci_protocol_audit_parser.add_argument("--require-report-policy", action="store_true")
    ci_protocol_audit_parser.add_argument("--output", default="artifacts/ci/ci_protocol_audit.json")
    ci_protocol_audit_parser.add_argument("--strict", action="store_true")
    ci_protocol_audit_parser.add_argument("--json", action="store_true")

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

    submit_slurm_plan_parser = subparsers.add_parser(
        "submit-slurm-plan",
        help="Submit Slurm jobs with enforced dependency topology",
    )
    submit_slurm_plan_parser.add_argument("--slurm-dir", default="maxionbench/orchestration/slurm")
    submit_slurm_plan_parser.add_argument("--seed", type=int, default=42)
    submit_slurm_plan_parser.add_argument("--skip-gpu", action="store_true")
    submit_slurm_plan_parser.add_argument("--dry-run", action="store_true")
    submit_slurm_plan_parser.add_argument("--json", action="store_true")

    verify_promotion_gate_parser = subparsers.add_parser(
        "verify-promotion-gate",
        help="Verify strict-readiness artifact before promotion",
    )
    verify_promotion_gate_parser.add_argument(
        "--strict-readiness-summary",
        default="artifacts/conformance_strict/engine_readiness_summary.json",
    )
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

    conformance_matrix_parser = subparsers.add_parser(
        "conformance-matrix",
        help="Run conformance tests for all adapter configs",
    )
    conformance_matrix_parser.add_argument("--config-dir", default="configs/conformance")
    conformance_matrix_parser.add_argument("--out-dir", default="artifacts/conformance")
    conformance_matrix_parser.add_argument("--timeout-s", type=float, default=300.0)

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
        if args.json:
            verify_argv.append("--json")
        return verify_pins_main(verify_argv)
    if args.command == "verify-dataset-manifests":
        from maxionbench.tools.verify_dataset_manifests import main as verify_dataset_manifests_main

        verify_argv = ["--manifest-dir", args.manifest_dir]
        if args.json:
            verify_argv.append("--json")
        return verify_dataset_manifests_main(verify_argv)
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
    if args.command == "verify-slurm-plan":
        from maxionbench.tools.verify_slurm_plan import main as verify_slurm_plan_main

        verify_argv: list[str] = ["--slurm-dir", args.slurm_dir]
        if args.skip_gpu:
            verify_argv.append("--skip-gpu")
        if args.json:
            verify_argv.append("--json")
        return verify_slurm_plan_main(verify_argv)
    if args.command == "validate-slurm-snapshots":
        from maxionbench.tools.validate_slurm_snapshots import main as validate_slurm_snapshots_main

        validate_argv: list[str] = ["--required-baseline-scenario", args.required_baseline_scenario]
        for path in args.verify_paths or []:
            validate_argv.extend(["--verify-path", path])
        for path in args.submit_paths or []:
            validate_argv.extend(["--submit-path", path])
        if args.json:
            validate_argv.append("--json")
        return validate_slurm_snapshots_main(validate_argv)
    if args.command == "ci-protocol-audit":
        from maxionbench.tools.ci_protocol_audit import main as ci_protocol_audit_main

        audit_argv: list[str] = [
            "--config-dir",
            args.config_dir,
            "--slurm-dir",
            args.slurm_dir,
            "--manifest-dir",
            args.manifest_dir,
            "--required-baseline-scenario",
            args.required_baseline_scenario,
            "--output",
            args.output,
        ]
        for path in args.verify_paths or []:
            audit_argv.extend(["--verify-path", path])
        for path in args.submit_paths or []:
            audit_argv.extend(["--submit-path", path])
        if args.report_input:
            audit_argv.extend(["--report-input", args.report_input])
        if args.require_report_policy:
            audit_argv.append("--require-report-policy")
        if args.strict:
            audit_argv.append("--strict")
        if args.json:
            audit_argv.append("--json")
        return ci_protocol_audit_main(audit_argv)
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
    if args.command == "submit-slurm-plan":
        from maxionbench.orchestration.slurm.submit_plan import main as submit_slurm_plan_main

        submit_argv: list[str] = ["--slurm-dir", args.slurm_dir, "--seed", str(args.seed)]
        if args.skip_gpu:
            submit_argv.append("--skip-gpu")
        if args.dry_run:
            submit_argv.append("--dry-run")
        if args.json:
            submit_argv.append("--json")
        return submit_slurm_plan_main(submit_argv)
    if args.command == "verify-promotion-gate":
        from maxionbench.tools.verify_promotion_gate import main as verify_promotion_gate_main

        verify_argv: list[str] = ["--strict-readiness-summary", args.strict_readiness_summary]
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
    if args.command == "conformance-matrix":
        from maxionbench.conformance.matrix import main as conformance_matrix_main

        return conformance_matrix_main(
            [
                "--config-dir",
                args.config_dir,
                "--out-dir",
                args.out_dir,
                "--timeout-s",
                str(args.timeout_s),
            ]
        )
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
