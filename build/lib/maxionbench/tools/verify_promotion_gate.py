"""Verify strict-readiness artifact before result-bundle promotion."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys
from typing import Any

REQUIRED_ADAPTERS = (
    "qdrant",
    "milvus",
    "weaviate",
    "opensearch",
    "pgvector",
    "lancedb-service",
    "lancedb-inproc",
    "faiss-cpu",
    "faiss-gpu",
)


def _expected_required_adapters(*, allow_gpu_unavailable: bool) -> list[str]:
    expected = list(REQUIRED_ADAPTERS)
    if allow_gpu_unavailable:
        expected = [name for name in expected if name != "faiss-gpu"]
    return expected


def verify_promotion_gate(
    *,
    strict_readiness_summary_path: Path,
    conformance_matrix_path: Path | None = None,
) -> dict[str, Any]:
    path = strict_readiness_summary_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strict readiness summary not found: {path}")
    resolved_matrix_path: Path | None = None
    if conformance_matrix_path is not None:
        resolved_matrix_path = conformance_matrix_path.resolve()
        if not resolved_matrix_path.exists():
            raise FileNotFoundError(f"Conformance matrix not found: {resolved_matrix_path}")

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
    allow_gpu_unavailable_raw = payload.get("allow_gpu_unavailable")
    allow_gpu_unavailable = (
        allow_gpu_unavailable_raw if isinstance(allow_gpu_unavailable_raw, bool) else None
    )
    require_mock_pass_raw = payload.get("require_mock_pass")
    require_mock_pass = require_mock_pass_raw if isinstance(require_mock_pass_raw, bool) else None
    error_count = payload.get("error_count", 0)
    conformance_rows = payload.get("conformance_rows", 0)
    conformance_status_counts_raw = payload.get("conformance_status_counts")
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
    if allow_gpu_unavailable is None:
        reasons.append("strict readiness summary missing boolean `allow_gpu_unavailable` field")
    if isinstance(required_adapters, list):
        if any(not isinstance(item, str) for item in required_adapters):
            reasons.append("strict readiness summary required_adapters must contain only strings")
        actual_required = [str(item) for item in required_adapters]
        if len(actual_required) != len(set(actual_required)):
            reasons.append("strict readiness summary required_adapters contains duplicates")
        if allow_gpu_unavailable is not None and all(isinstance(item, str) for item in required_adapters):
            expected_required = _expected_required_adapters(allow_gpu_unavailable=allow_gpu_unavailable)
            missing = sorted(set(expected_required) - set(actual_required))
            extra = sorted(set(actual_required) - set(expected_required))
            if missing or extra:
                reasons.append(
                    "strict readiness summary required_adapters mismatch "
                    f"(missing={missing}, extra={extra})"
                )
    if allow_nonpass_status is None:
        reasons.append("strict readiness summary missing boolean `allow_nonpass_status` field")
    elif allow_nonpass_status:
        reasons.append("strict readiness summary was generated with allow_nonpass_status=true")
    if require_mock_pass is None:
        reasons.append("strict readiness summary missing boolean `require_mock_pass` field")
    elif not require_mock_pass:
        reasons.append("strict readiness summary was generated with require_mock_pass=false")
    if behavior_cards_ok is None:
        reasons.append("strict readiness summary missing boolean `behavior_cards_ok` field")
    elif not behavior_cards_ok:
        reasons.append("strict readiness summary reports behavior_cards_ok=false")
    if not isinstance(errors_field, list):
        reasons.append("strict readiness summary missing list `errors` field")
    elif errors_field:
        reasons.append("strict readiness summary errors list is not empty")
    if not isinstance(conformance_status_counts_raw, dict):
        reasons.append("strict readiness summary missing mapping `conformance_status_counts` field")
    else:
        status_counts: dict[str, int] = {}
        invalid_count_payload = False
        for key, value in conformance_status_counts_raw.items():
            try:
                status_counts[str(key)] = int(value)
            except (TypeError, ValueError):
                invalid_count_payload = True
                break
        if invalid_count_payload:
            reasons.append("strict readiness summary has non-integer conformance_status_counts values")
        else:
            non_pass_counts = {k: v for k, v in status_counts.items() if k != "pass" and v > 0}
            allow_faiss_gpu_nonpass = bool(allow_gpu_unavailable) and (resolved_matrix_path is not None)
            if non_pass_counts and bool(allow_gpu_unavailable) and (resolved_matrix_path is None):
                reasons.append(
                    "strict readiness summary has non-pass conformance_status_counts in allow_gpu_unavailable mode; "
                    "provide --conformance-matrix to verify faiss-gpu-only non-pass rows"
                )
            elif non_pass_counts and not allow_faiss_gpu_nonpass:
                reasons.append(
                    "strict readiness summary conformance_status_counts includes non-pass rows "
                    f"{dict(sorted(non_pass_counts.items()))}"
                )
            pass_count = int(status_counts.get("pass", 0))
            if pass_count < 1:
                reasons.append("strict readiness summary conformance_status_counts.pass must be >= 1")
            try:
                conformance_rows_int_for_counts = int(conformance_rows)
            except (TypeError, ValueError):
                conformance_rows_int_for_counts = -1
            if conformance_rows_int_for_counts >= 0 and sum(status_counts.values()) != conformance_rows_int_for_counts:
                reasons.append(
                    "strict readiness summary conformance row count mismatch with conformance_status_counts "
                    f"(rows={conformance_rows_int_for_counts}, counts_sum={sum(status_counts.values())})"
                )
    try:
        conformance_rows_int = int(conformance_rows)
    except (TypeError, ValueError):
        conformance_rows_int = 0
    if conformance_rows_int < 1:
        reasons.append("strict readiness summary has no conformance rows")
    if isinstance(required_adapters, list):
        required_adapter_count = len(required_adapters)
        if conformance_rows_int < required_adapter_count:
            reasons.append(
                "strict readiness summary has fewer conformance rows than required adapters "
                f"(rows={conformance_rows_int}, required_adapters={required_adapter_count})"
            )
    try:
        error_count_int = int(error_count)
    except (TypeError, ValueError):
        error_count_int = 1
    if error_count_int != 0:
        reasons.append(f"strict readiness summary error_count={error_count_int} (expected 0)")

    matrix_rows: int | None = None
    matrix_status_counts: dict[str, int] | None = None
    matrix_adapter_counts: dict[str, int] | None = None
    matrix_adapter_status_counts: dict[str, dict[str, int]] | None = None
    if resolved_matrix_path is not None:
        (
            matrix_rows,
            matrix_status_counts,
            matrix_adapter_counts,
            matrix_adapter_status_counts,
        ) = _read_matrix_status_counts(resolved_matrix_path)
        if matrix_rows != conformance_rows_int:
            reasons.append(
                "strict readiness summary conformance_rows disagrees with conformance matrix "
                f"(summary={conformance_rows_int}, matrix={matrix_rows})"
            )
        if isinstance(conformance_status_counts_raw, dict):
            summary_counts: dict[str, int] = {}
            valid = True
            for key, value in conformance_status_counts_raw.items():
                try:
                    summary_counts[str(key)] = int(value)
                except (TypeError, ValueError):
                    valid = False
                    break
            if valid and summary_counts != matrix_status_counts:
                reasons.append(
                    "strict readiness summary conformance_status_counts disagrees with conformance matrix "
                    f"(summary={dict(sorted(summary_counts.items()))}, matrix={dict(sorted(matrix_status_counts.items()))})"
                )
        if isinstance(required_adapters, list) and all(isinstance(item, str) for item in required_adapters):
            observed = set(matrix_adapter_counts or {})
            missing_from_matrix = sorted(set(required_adapters) - observed)
            if missing_from_matrix:
                reasons.append(
                    "strict readiness summary required_adapters missing in conformance matrix "
                    f"{missing_from_matrix}"
                )
        if bool(allow_gpu_unavailable):
            disallowed_non_pass: dict[str, dict[str, int]] = {}
            for adapter, per_status in (matrix_adapter_status_counts or {}).items():
                if adapter == "faiss-gpu":
                    continue
                non_pass_for_adapter = {k: v for k, v in per_status.items() if k != "pass" and v > 0}
                if non_pass_for_adapter:
                    disallowed_non_pass[adapter] = dict(sorted(non_pass_for_adapter.items()))
            if disallowed_non_pass:
                reasons.append(
                    "allow_gpu_unavailable mode permits non-pass matrix rows only for `faiss-gpu`; "
                    f"found non-pass rows for other adapters {dict(sorted(disallowed_non_pass.items()))}"
                )
        if require_mock_pass:
            mock_status_counts = (matrix_adapter_status_counts or {}).get("mock", {})
            mock_pass_rows = int(mock_status_counts.get("pass", 0))
            if mock_pass_rows < 1:
                reasons.append(
                    "strict readiness summary requires mock pass, but conformance matrix has no `mock` pass row"
                )

    return {
        "strict_readiness_summary_path": str(path),
        "conformance_matrix_path": str(resolved_matrix_path) if resolved_matrix_path is not None else None,
        "ready_for_promotion": len(reasons) == 0,
        "reasons": reasons,
        "strict_readiness_checks": {
            "pass_field": isinstance(pass_raw, bool),
            "allow_gpu_unavailable_field": allow_gpu_unavailable is not None,
            "allow_gpu_unavailable": allow_gpu_unavailable,
            "allow_nonpass_status_field": allow_nonpass_status is not None,
            "allow_nonpass_status": allow_nonpass_status,
            "require_mock_pass_field": require_mock_pass is not None,
            "require_mock_pass": require_mock_pass,
            "conformance_status_counts_field": isinstance(conformance_status_counts_raw, dict),
            "conformance_matrix_checked": resolved_matrix_path is not None,
            "behavior_cards_ok_field": behavior_cards_ok is not None,
            "behavior_cards_ok": behavior_cards_ok,
        },
        "matrix_observed": {
            "rows": matrix_rows,
            "status_counts": dict(sorted((matrix_status_counts or {}).items())) if matrix_status_counts is not None else None,
            "adapter_counts": dict(sorted((matrix_adapter_counts or {}).items())) if matrix_adapter_counts is not None else None,
            "adapter_status_counts": (
                {
                    adapter: dict(sorted(status_counts.items()))
                    for adapter, status_counts in sorted((matrix_adapter_status_counts or {}).items())
                }
                if matrix_adapter_status_counts is not None
                else None
            ),
        },
        "summary": payload,
        "pass": len(reasons) == 0,
    }


def _read_matrix_status_counts(
    path: Path,
) -> tuple[int, dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "adapter" not in fieldnames or "status" not in fieldnames:
            raise ValueError("Conformance matrix must include `adapter` and `status` columns")
        rows = 0
        status_counts: dict[str, int] = {}
        adapter_counts: dict[str, int] = {}
        adapter_status_counts: dict[str, dict[str, int]] = {}
        for row_index, row in enumerate(reader, start=2):
            rows += 1
            adapter = str((row.get("adapter") if isinstance(row, dict) else "") or "")
            adapter_key = adapter.strip()
            if not adapter_key:
                raise ValueError(f"Conformance matrix row {row_index} has empty `adapter` value")
            adapter_counts[adapter_key] = adapter_counts.get(adapter_key, 0) + 1
            status = str((row.get("status") if isinstance(row, dict) else "") or "")
            key = status.strip()
            if not key:
                raise ValueError(f"Conformance matrix row {row_index} has empty `status` value")
            status_counts[key] = status_counts.get(key, 0) + 1
            per_adapter = adapter_status_counts.setdefault(adapter_key, {})
            per_adapter[key] = per_adapter.get(key, 0) + 1
    return rows, status_counts, adapter_counts, adapter_status_counts


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify strict-readiness artifact for promotion gate")
    parser.add_argument(
        "--strict-readiness-summary",
        default="artifacts/conformance_strict/engine_readiness_summary.json",
        help="Path to strict readiness summary JSON artifact",
    )
    parser.add_argument(
        "--conformance-matrix",
        default=None,
        help="Optional conformance_matrix.csv path to cross-check strict summary provenance",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = verify_promotion_gate(
            strict_readiness_summary_path=Path(args.strict_readiness_summary),
            conformance_matrix_path=Path(args.conformance_matrix) if args.conformance_matrix else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"verify-promotion-gate failed: {exc}", file=sys.stderr)
        return 2
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
