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

    validate_parser = subparsers.add_parser("validate", help="Validate output artifacts")
    validate_parser.add_argument("--input", required=True)
    validate_mode_group = validate_parser.add_mutually_exclusive_group()
    validate_mode_group.add_argument("--strict-schema", action="store_true")
    validate_mode_group.add_argument("--legacy-ok", action="store_true")
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
    verify_branch_parser.add_argument("--json", action="store_true")

    verify_pins_parser = subparsers.add_parser(
        "verify-pins",
        help="Verify pinned scenario config values",
    )
    verify_pins_parser.add_argument("--config-dir", default="configs/scenarios")
    verify_pins_parser.add_argument("--json", action="store_true")

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
        return run_main(run_argv)
    if args.command == "validate":
        from maxionbench.tools.validate_outputs import validate_path
        import json

        summary = validate_path(
            Path(args.input).resolve(),
            strict_schema=not bool(args.legacy_ok),
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
        if args.json:
            verify_argv.append("--json")
        return verify_branch_main(verify_argv)
    if args.command == "verify-pins":
        from maxionbench.tools.verify_pins import main as verify_pins_main

        verify_argv: list[str] = ["--config-dir", args.config_dir]
        if args.json:
            verify_argv.append("--json")
        return verify_pins_main(verify_argv)
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
