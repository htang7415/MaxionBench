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

    pass_raw = payload.get("pass")
    pass_value = pass_raw if isinstance(pass_raw, bool) else False
    required_adapters = payload.get("required_adapters")
    allow_nonpass_status_raw = payload.get("allow_nonpass_status")
    allow_nonpass_status = (
        allow_nonpass_status_raw if isinstance(allow_nonpass_status_raw, bool) else None
    )
    error_count = payload.get("error_count", 0)
    conformance_rows = payload.get("conformance_rows", 0)
    behavior_cards_ok_raw = payload.get("behavior_cards_ok")
    behavior_cards_ok = behavior_cards_ok_raw if isinstance(behavior_cards_ok_raw, bool) else None
    errors_field = payload.get("errors")

    reasons: list[str] = []
    if not isinstance(pass_raw, bool):
        reasons.append("strict readiness summary missing boolean `pass` field")
    if not pass_value:
        reasons.append("strict readiness summary reports pass=false")
    if not isinstance(required_adapters, list) or not required_adapters:
        reasons.append("strict readiness summary missing required_adapters")
    if allow_nonpass_status is None:
        reasons.append("strict readiness summary missing boolean `allow_nonpass_status` field")
    elif allow_nonpass_status:
        reasons.append("strict readiness summary was generated with allow_nonpass_status=true")
    if behavior_cards_ok is None:
        reasons.append("strict readiness summary missing boolean `behavior_cards_ok` field")
    elif not behavior_cards_ok:
        reasons.append("strict readiness summary reports behavior_cards_ok=false")
    if not isinstance(errors_field, list):
        reasons.append("strict readiness summary missing list `errors` field")
    elif errors_field:
        reasons.append("strict readiness summary errors list is not empty")
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
        "strict_readiness_checks": {
            "pass_field": isinstance(pass_raw, bool),
            "allow_nonpass_status_field": allow_nonpass_status is not None,
            "allow_nonpass_status": allow_nonpass_status,
            "behavior_cards_ok_field": behavior_cards_ok is not None,
            "behavior_cards_ok": behavior_cards_ok,
        },
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
