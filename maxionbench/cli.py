"""Top-level CLI for MaxionBench."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from maxionbench.conformance.run import main as conformance_main


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

    conformance_parser = subparsers.add_parser("conformance", help="Run adapter conformance tests")
    conformance_parser.add_argument("--adapter", default="mock")
    conformance_parser.add_argument("--adapter-options-json", default="{}")
    conformance_parser.add_argument("--collection", default="conformance")
    conformance_parser.add_argument("--dimension", type=int, default=4)
    conformance_parser.add_argument("--metric", default="ip")

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
        from maxionbench.tools.validate_outputs import validate_run_directory

        validate_run_directory(Path(args.input).resolve())
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
    raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
