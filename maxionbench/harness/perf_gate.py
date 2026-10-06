"""CI performance gate: check result bundles against committed per-metric bounds.

Bounds apply to the cell mean of each metric (over repeats); `cells` selects cells by their
matrix parameters, and a bound without `cells` applies to every cell. Any violation, a missing
metric, a failed trial, or an experiment without a result fails the gate.

    python -m maxionbench.harness.perf_gate ci/perf_baseline.yaml artifacts/ci/*/results.json
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


def check(baseline: Mapping[str, Any], results: list[Mapping[str, Any]]) -> list[str]:
    by_name = {r["name"]: r for r in results}
    problems = []
    for name, rules in baseline["experiments"].items():
        result = by_name.get(name)
        if result is None:
            problems.append(f"{name}: no result bundle")
            continue
        failed = [t["trial_id"] for t in result["trials"] if t["status"] != "ok"]
        if failed:
            problems.append(f"{name}: failed trials {failed}")
        for rule in rules:
            selector = {str(k): str(v) for k, v in (rule.get("cells") or {}).items()}
            cells = [c for c in result["cells"] if all(str(c["params"].get(k)) == v for k, v in selector.items())]
            if not cells:
                problems.append(f"{name}: no cell matches {selector}")
            for cell in cells:
                where = f"{name}[{cell['cell_id']}]"
                for metric, bound in rule["metrics"].items():
                    stat = cell["metrics"].get(metric)
                    if stat is None:
                        problems.append(f"{where}: missing metric {metric}")
                        continue
                    value = stat["mean"]
                    if "min" in bound and value < bound["min"]:
                        problems.append(f"{where}: {metric} {value:.4g} < min {bound['min']}")
                    if "max" in bound and value > bound["max"]:
                        problems.append(f"{where}: {metric} {value:.4g} > max {bound['max']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)
    baseline = yaml.safe_load(args.baseline.read_text(encoding="utf-8"))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    problems = check(baseline, results)
    for r in results:
        for cell in r["cells"]:
            summary = {m: round(cell["metrics"][m]["mean"], 2) for m in sorted(cell["metrics"])
                       if m in ("errors", "ok", "ttft_p50_ms", "tpot_p50_ms", "output_tokens_per_s", "slo_attainment")}
            print(f"{r['name']}[{cell['cell_id']}] {summary}")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    print("perf gate: " + ("FAILED" if problems else "passed"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
