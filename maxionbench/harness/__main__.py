"""CLI: python -m maxionbench.harness {run,schema,compare}."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from maxionbench.harness.compare import DEFAULT_METRICS, compare, load_result
from maxionbench.harness.runner import run_experiment
from maxionbench.harness.schema_export import SCHEMA_PATH, schema_text
from maxionbench.harness.spec import load_spec


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="python -m maxionbench.harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run an experiment spec")
    run_p.add_argument("spec")
    run_p.add_argument("--out", default="artifacts/harness")
    schema_p = sub.add_parser("schema", help="print, write, or check the result JSON Schema")
    mode = schema_p.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    cmp_p = sub.add_parser("compare", help="CI overlap between two result bundles")
    cmp_p.add_argument("a")
    cmp_p.add_argument("b")
    cmp_p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args(argv)

    if args.command == "run":
        out_dir, result = run_experiment(load_spec(Path(args.spec)), Path(args.out))
        failed = sum(1 for t in result.trials if t.status != "ok")
        print(json.dumps({"out_dir": str(out_dir), "trials": len(result.trials), "failed": failed}, indent=2))
        return 1 if failed else 0
    if args.command == "schema":
        text = schema_text()
        if args.write:
            SCHEMA_PATH.write_text(text, encoding="utf-8")
        elif args.check:
            if not SCHEMA_PATH.exists() or SCHEMA_PATH.read_text(encoding="utf-8") != text:
                print(f"{SCHEMA_PATH} is stale; run: python -m maxionbench.harness schema --write", file=sys.stderr)
                return 1
        else:
            sys.stdout.write(text)
        return 0
    report = compare(load_result(Path(args.a)), load_result(Path(args.b)), args.metrics.split(","))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
