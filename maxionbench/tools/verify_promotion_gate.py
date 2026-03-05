"""Verify strict-readiness artifact before result-bundle promotion."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any


def verify_promotion_gate(
    *,
    strict_readiness_summary_path: Path,
) -> dict[str, Any]:
    path = strict_readiness_summary_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strict readiness summary not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Strict readiness summary must be a JSON object")

    pass_value = bool(payload.get("pass", False))
    required_adapters = payload.get("required_adapters")
    error_count = payload.get("error_count", 0)
    conformance_rows = payload.get("conformance_rows", 0)

    reasons: list[str] = []
    if not pass_value:
        reasons.append("strict readiness summary reports pass=false")
    if not isinstance(required_adapters, list) or not required_adapters:
        reasons.append("strict readiness summary missing required_adapters")
    try:
        conformance_rows_int = int(conformance_rows)
    except (TypeError, ValueError):
        conformance_rows_int = 0
    if conformance_rows_int < 1:
        reasons.append("strict readiness summary has no conformance rows")
    try:
        error_count_int = int(error_count)
    except (TypeError, ValueError):
        error_count_int = 1
    if error_count_int != 0:
        reasons.append(f"strict readiness summary error_count={error_count_int} (expected 0)")

    return {
        "strict_readiness_summary_path": str(path),
        "ready_for_promotion": len(reasons) == 0,
        "reasons": reasons,
        "summary": payload,
        "pass": len(reasons) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify strict-readiness artifact for promotion gate")
    parser.add_argument(
        "--strict-readiness-summary",
        default="artifacts/conformance_strict/engine_readiness_summary.json",
        help="Path to strict readiness summary JSON artifact",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_promotion_gate(strict_readiness_summary_path=Path(args.strict_readiness_summary))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print("promotion gate passed")
        else:
            print("promotion gate failed")
            for reason in summary["reasons"]:
                print(f"- {reason}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
