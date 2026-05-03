"""Verify strict-readiness artifact before result-bundle promotion."""

from __future__ import annotations

from argparse import ArgumentParser, SUPPRESS
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from maxionbench.schemas.result_schema import RUN_STATUS_FILENAME, read_run_status

REQUIRED_ADAPTERS = (
    "qdrant",
    "pgvector",
    "lancedb-service",
    "lancedb-inproc",
    "faiss-cpu",
)

PORTABLE_SCENARIOS = ("s1_single_hop", "s2_streaming_memory", "s3_multi_hop")
PORTABLE_PROMOTION_RULES: dict[str, dict[str, Any]] = {
    "b0": {
        "to_budget": "b1",
        "quality_thresholds": {
            "s1_single_hop": 0.1875,
            "s2_streaming_memory": 0.1875,
            "s3_multi_hop": 0.225,
        },
        "s2_freshness_hit_at_5s": 0.6,
        "max_error_rate": 0.05,
        "prune_survivors": False,
    },
    "b1": {
        "to_budget": "b2",
        "quality_thresholds": {
            "s1_single_hop": 0.225,
            "s2_streaming_memory": 0.225,
            "s3_multi_hop": 0.27,
        },
        "s2_freshness_hit_at_5s": 0.8,
        "max_error_rate": 0.05,
        "prune_survivors": True,
        "max_survivors_per_scenario": 3,
    },
}


def _expected_required_adapters(*, allow_gpu_unavailable: bool) -> list[str]:
    del allow_gpu_unavailable
    return list(REQUIRED_ADAPTERS)


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
            if non_pass_counts:
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


def verify_portable_promotion_gate(
    *,
    results_path: Path,
    from_budget: str,
    out_candidates_path: Path | None = None,
) -> dict[str, Any]:
    budget = str(from_budget).strip().lower()
    if budget not in PORTABLE_PROMOTION_RULES:
        raise ValueError(f"from_budget must be one of {sorted(PORTABLE_PROMOTION_RULES)}, got {from_budget!r}")
    path = results_path.expanduser().resolve()
    frame = _load_results_frame(path)
    rule = PORTABLE_PROMOTION_RULES[budget]
    rows = [
        row
        for row in (_portable_candidate_from_row(row) for row in frame.to_dict(orient="records"))
        if row is not None and row["budget_level"] == budget
    ]

    survivors: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for row in rows:
        reject_reasons = _portable_rejection_reasons(row=row, rule=rule)
        if reject_reasons:
            rejected = dict(row)
            rejected["reasons"] = reject_reasons
            rejections.append(rejected)
        else:
            survivors.append(dict(row))

    promoted = _prune_portable_survivors(survivors=survivors, rule=rule)
    observed_scenarios = sorted({str(row["scenario"]) for row in rows})
    survivor_scenarios = sorted({str(row["scenario"]) for row in promoted})
    missing_scenarios = [scenario for scenario in PORTABLE_SCENARIOS if scenario not in observed_scenarios]
    missing_survivors = [scenario for scenario in PORTABLE_SCENARIOS if scenario not in survivor_scenarios]

    reasons: list[str] = []
    if not rows:
        reasons.append(f"no MaxionBench result rows found for budget {budget}")
    if missing_scenarios:
        reasons.append(f"missing portable scenarios for budget {budget}: {missing_scenarios}")
    if missing_survivors:
        reasons.append(f"no promoted survivor for portable scenarios: {missing_survivors}")

    summary = {
        "mode": "maxionbench-promotion",
        "results_path": str(path),
        "from_budget": budget,
        "to_budget": str(rule["to_budget"]),
        "thresholds": {
            "quality": dict(rule["quality_thresholds"]),
            "s2_freshness_hit_at_5s": float(rule["s2_freshness_hit_at_5s"]),
            "max_error_rate": float(rule["max_error_rate"]),
        },
        "prune_survivors": bool(rule.get("prune_survivors", False)),
        "max_survivors_per_scenario": rule.get("max_survivors_per_scenario"),
        "rows_checked": len(rows),
        "survivor_count_before_prune": len(survivors),
        "promoted_survivor_count": len(promoted),
        "rejected_count": len(rejections),
        "observed_scenarios": observed_scenarios,
        "missing_scenarios": missing_scenarios,
        "missing_survivors": missing_survivors,
        "survivors": promoted,
        "rejections": rejections,
        "ready_for_promotion": len(reasons) == 0,
        "reasons": reasons,
        "pass": len(reasons) == 0,
    }
    if out_candidates_path is not None:
        out_path = out_candidates_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["out_candidates_path"] = str(out_path)
    return summary


def _load_results_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Portable results path not found: {path}")
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Portable results file must be a .parquet file: {path}")
        return pd.read_parquet(path)

    frames: list[pd.DataFrame] = []
    for result_path in sorted(path.rglob("results.parquet")):
        status_path = result_path.parent / RUN_STATUS_FILENAME
        if status_path.exists():
            status = str(read_run_status(status_path).get("status") or "").strip().lower()
            if status != "success":
                continue
        frame = pd.read_parquet(result_path)
        frame["__run_path"] = str(result_path.parent)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _portable_candidate_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    scenario = str(row.get("scenario") or "").strip()
    payload = _json_mapping(row.get("search_params_json"))
    profile = str(payload.get("profile") or row.get("profile") or row.get("__meta_profile") or "").strip()
    if scenario not in PORTABLE_SCENARIOS and profile not in {"maxionbench", "portable-agentic"}:
        return None
    budget_level = str(payload.get("budget_level") or row.get("budget_level") or row.get("__meta_budget_level") or "").strip().lower()
    embedding_model = str(payload.get("embedding_model") or row.get("embedding_model") or row.get("__meta_embedding_model") or "")
    clients_read = _safe_int(row.get("clients_read"), default=0)
    quality = _portable_quality_value(scenario=scenario, row=row, payload=payload)
    freshness = _safe_float(payload.get("freshness_hit_at_5s", row.get("freshness_hit_at_5s")))
    errors = _safe_int(row.get("errors"), default=0)
    measured_requests = _safe_int(row.get("measure_requests", payload.get("measured_requests")), default=0)
    error_rate = float(errors) / max(float(measured_requests), 1.0)
    task_cost = _safe_float(payload.get("task_cost_est", row.get("task_cost_est")))
    p99_ms = _safe_float(row.get("p99_ms"))
    qps = _safe_float(row.get("qps"))
    candidate = {
        "run_id": str(row.get("run_id") or ""),
        "scenario": scenario,
        "budget_level": budget_level,
        "engine": str(row.get("engine") or ""),
        "embedding_model": embedding_model,
        "clients_read": clients_read,
        "clients_write": _safe_int(row.get("clients_write"), default=0),
        "quality_target": _safe_float(row.get("quality_target")),
        "primary_quality_metric": str(payload.get("primary_quality_metric") or _portable_quality_metric(scenario)),
        "primary_quality_value": quality,
        "freshness_hit_at_5s": freshness,
        "error_rate": error_rate,
        "errors": errors,
        "measure_requests": measured_requests,
        "task_cost_est": task_cost,
        "p99_ms": p99_ms,
        "qps": qps,
    }
    candidate["config_key"] = (
        f"{candidate['scenario']}|{candidate['engine']}|{candidate['embedding_model']}|"
        f"clients={candidate['clients_read']}|target={candidate['quality_target']}"
    )
    return candidate


def _portable_quality_value(*, scenario: str, row: dict[str, Any], payload: dict[str, Any]) -> float:
    if scenario == "s3_multi_hop":
        return _first_finite(
            payload.get("evidence_coverage_at_10"),
            payload.get("primary_quality_value"),
            row.get("evidence_coverage_at_10"),
            row.get("ndcg_at_10"),
        )
    return _first_finite(payload.get("primary_quality_value"), row.get("ndcg_at_10"))


def _portable_quality_metric(scenario: str) -> str:
    if scenario == "s3_multi_hop":
        return "evidence_coverage@10"
    return "ndcg_at_10"


def _portable_rejection_reasons(*, row: dict[str, Any], rule: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    scenario = str(row["scenario"])
    threshold = float(rule["quality_thresholds"].get(scenario, math.inf))
    quality = _safe_float(row.get("primary_quality_value"))
    if not _finite_at_least(quality, threshold):
        reasons.append(f"{_portable_quality_metric(scenario)}={quality} below {threshold}")
    error_rate = _safe_float(row.get("error_rate"))
    max_error_rate = float(rule["max_error_rate"])
    if not math.isfinite(error_rate) or error_rate > max_error_rate:
        reasons.append(f"error_rate={error_rate} above {max_error_rate}")
    if scenario == "s2_streaming_memory":
        freshness = _safe_float(row.get("freshness_hit_at_5s"))
        freshness_threshold = float(rule["s2_freshness_hit_at_5s"])
        if not _finite_at_least(freshness, freshness_threshold):
            reasons.append(f"freshness_hit_at_5s={freshness} below {freshness_threshold}")
    return reasons


def _prune_portable_survivors(*, survivors: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        survivors,
        key=lambda row: (
            str(row["scenario"]),
            _sortable_float(row.get("task_cost_est")),
            _sortable_float(row.get("p99_ms")),
            -_sortable_float(row.get("qps")),
            str(row.get("engine") or ""),
            str(row.get("embedding_model") or ""),
            int(row.get("clients_read") or 0),
        ),
    )
    if not bool(rule.get("prune_survivors", False)):
        return ordered
    limit = int(rule.get("max_survivors_per_scenario", 3))
    counts: dict[str, int] = {}
    pruned: list[dict[str, Any]] = []
    for row in ordered:
        scenario = str(row["scenario"])
        count = counts.get(scenario, 0)
        if count >= limit:
            continue
        counts[scenario] = count + 1
        pruned.append(row)
    return pruned


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _first_finite(*values: Any) -> float:
    for value in values:
        candidate = _safe_float(value)
        if math.isfinite(candidate):
            return candidate
    return float("nan")


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_at_least(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value >= threshold


def _sortable_float(value: Any) -> float:
    candidate = _safe_float(value)
    return candidate if math.isfinite(candidate) else math.inf


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
        "--maxionbench-results",
        default=None,
        help="Directory or results.parquet file to check against MaxionBench B0/B1 promotion rules",
    )
    parser.add_argument("--portable-results", dest="maxionbench_results", default=None, help=SUPPRESS)
    parser.add_argument("--from-budget", default=None, choices=sorted(PORTABLE_PROMOTION_RULES))
    parser.add_argument("--out-candidates", default=None, help="Optional JSON path for promoted MaxionBench candidates")
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
        if args.maxionbench_results:
            if args.from_budget is None:
                parser.error("--from-budget is required with --maxionbench-results")
            summary = verify_portable_promotion_gate(
                results_path=Path(args.maxionbench_results),
                from_budget=str(args.from_budget),
                out_candidates_path=Path(args.out_candidates) if args.out_candidates else None,
            )
        else:
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
