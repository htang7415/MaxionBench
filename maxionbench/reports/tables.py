"""Tabular report exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_tables(
    *,
    frame: pd.DataFrame,
    out_dir: Path,
    mode: str,
    output_policy: dict[str, Any] | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    summary = _summary_table(frame)
    summary_path = out_dir / f"{mode}_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    t1 = _table_t1_env_runtime(frame)
    t1_path = out_dir / "T1_environment_runtime_pinning.csv"
    t1.to_csv(t1_path, index=False)
    outputs.append(t1_path)

    t2 = _table_t2_matched_quality(frame)
    t2_path = out_dir / "T2_matched_quality_throughput_rhu.csv"
    t2.to_csv(t2_path, index=False)
    outputs.append(t2_path)

    t3 = _table_t3_robustness(frame)
    t3_path = out_dir / "T3_robustness_summary.csv"
    t3.to_csv(t3_path, index=False)
    outputs.append(t3_path)

    t4 = _table_t4_decision(frame)
    t4_path = out_dir / "T4_decision_table.csv"
    t4.to_csv(t4_path, index=False)
    outputs.append(t4_path)

    stats_payload = {
        "mode": mode,
        "rows": int(len(frame)),
        "engines": sorted({str(v) for v in frame.get("engine", pd.Series(dtype=str)).tolist()}),
        "scenarios": sorted({str(v) for v in frame.get("scenario", pd.Series(dtype=str)).tolist()}),
        "run_ids": sorted({str(v) for v in frame.get("run_id", pd.Series(dtype=str)).tolist() if str(v)}),
        "config_fingerprints": sorted(
            {str(v) for v in frame.get("__meta_config_fingerprint", pd.Series(dtype=str)).tolist() if str(v)}
        ),
        "output_policy": dict(output_policy or {}),
    }
    stats_path = out_dir / f"{mode}_summary.meta.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    outputs.append(stats_path)

    return outputs


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "engine",
                "rows",
                "recall_at_10_mean",
                "p99_ms_mean",
                "qps_mean",
                "rhu_h_mean",
                "errors_sum",
            ]
        )

    grouped = (
        frame.groupby(["scenario", "engine"], dropna=False, sort=True)
        .agg(
            rows=("run_id", "count"),
            recall_at_10_mean=("recall_at_10", "mean"),
            p99_ms_mean=("p99_ms", "mean"),
            qps_mean=("qps", "mean"),
            rhu_h_mean=("rhu_h", "mean"),
            errors_sum=("errors", "sum"),
        )
        .reset_index()
    )
    return grouped


def _table_t1_env_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "engine",
        "engine_version",
        "scenario",
        "dataset_bundle",
        "ground_truth_source",
        "ground_truth_metric",
        "ground_truth_k",
        "ground_truth_engine",
        "clients_read",
        "clients_write",
        "sla_threshold_ms",
        "rtt_baseline_ms_p50_mean",
        "rtt_baseline_ms_p99_mean",
        "resource_cpu_vcpu_median",
        "resource_gpu_count_median",
        "resource_ram_gib_median",
        "resource_disk_tb_median",
        "rhu_rate_median",
        "w_c",
        "w_g",
        "w_r",
        "w_d",
        "c_ref_vcpu",
        "g_ref_gpu",
        "r_ref_gib",
        "d_ref_tb",
        "config_fingerprint",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"engine", "engine_version", "scenario", "dataset_bundle", "clients_read", "clients_write", "sla_threshold_ms"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "__meta_config_fingerprint" in working.columns:
        working["config_fingerprint"] = working["__meta_config_fingerprint"]
    else:
        working["config_fingerprint"] = ""
    for src, dst in [
        ("resource_cpu_vcpu", "resource_cpu_vcpu"),
        ("resource_gpu_count", "resource_gpu_count"),
        ("resource_ram_gib", "resource_ram_gib"),
        ("resource_disk_tb", "resource_disk_tb"),
        ("rhu_rate", "rhu_rate"),
        ("__meta_w_c", "__meta_w_c"),
        ("__meta_w_g", "__meta_w_g"),
        ("__meta_w_r", "__meta_w_r"),
        ("__meta_w_d", "__meta_w_d"),
        ("__meta_c_ref_vcpu", "__meta_c_ref_vcpu"),
        ("__meta_g_ref_gpu", "__meta_g_ref_gpu"),
        ("__meta_r_ref_gib", "__meta_r_ref_gib"),
        ("__meta_d_ref_tb", "__meta_d_ref_tb"),
        ("__meta_ground_truth_source", "__meta_ground_truth_source"),
        ("__meta_ground_truth_metric", "__meta_ground_truth_metric"),
        ("__meta_ground_truth_engine", "__meta_ground_truth_engine"),
        ("__meta_ground_truth_k", "__meta_ground_truth_k"),
    ]:
        if src not in working.columns:
            if src.startswith("__meta_ground_truth_") and not src.endswith("_k"):
                working[src] = ""
            else:
                working[src] = float("nan")
        if dst != src:
            working[dst] = working[src]
    grouped = (
        working.groupby(
            [
                "engine",
                "engine_version",
                "scenario",
                "dataset_bundle",
                "__meta_ground_truth_source",
                "__meta_ground_truth_metric",
                "__meta_ground_truth_engine",
                "__meta_ground_truth_k",
                "clients_read",
                "clients_write",
                "sla_threshold_ms",
                "config_fingerprint",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            rtt_baseline_ms_p50_mean=("rtt_baseline_ms_p50", "mean"),
            rtt_baseline_ms_p99_mean=("rtt_baseline_ms_p99", "mean"),
            resource_cpu_vcpu_median=("resource_cpu_vcpu", "median"),
            resource_gpu_count_median=("resource_gpu_count", "median"),
            resource_ram_gib_median=("resource_ram_gib", "median"),
            resource_disk_tb_median=("resource_disk_tb", "median"),
            rhu_rate_median=("rhu_rate", "median"),
            w_c=("__meta_w_c", "first"),
            w_g=("__meta_w_g", "first"),
            w_r=("__meta_w_r", "first"),
            w_d=("__meta_w_d", "first"),
            c_ref_vcpu=("__meta_c_ref_vcpu", "first"),
            g_ref_gpu=("__meta_g_ref_gpu", "first"),
            r_ref_gib=("__meta_r_ref_gib", "first"),
            d_ref_tb=("__meta_d_ref_tb", "first"),
        )
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "__meta_ground_truth_source": "ground_truth_source",
            "__meta_ground_truth_metric": "ground_truth_metric",
            "__meta_ground_truth_engine": "ground_truth_engine",
            "__meta_ground_truth_k": "ground_truth_k",
        }
    )
    return grouped[columns]


def _table_t2_matched_quality(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "dataset_bundle",
        "engine",
        "quality_target",
        "throughput_qps_median",
        "p99_ms_median",
        "rhu_h_median",
        "rows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"scenario", "dataset_bundle", "engine", "quality_target", "qps", "p99_ms", "rhu_h"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.groupby(["scenario", "dataset_bundle", "engine", "quality_target"], dropna=False, sort=True)
        .agg(
            throughput_qps_median=("qps", "median"),
            p99_ms_median=("p99_ms", "median"),
            rhu_h_median=("rhu_h", "median"),
            rows=("run_id", "count"),
        )
        .reset_index()
    )
    return grouped[columns]


def _table_t3_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "dataset_bundle",
        "engine",
        "p99_ms_median",
        "p99_inflation_median",
        "p99_inflation_valid_rows",
        "p99_inflation_nan_rows",
        "p99_inflation_status",
        "sla_violation_rate_mean",
        "errors_sum",
        "rows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"scenario", "dataset_bundle", "engine", "p99_ms", "sla_violation_rate", "errors"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    working["__search_payload"] = working.get("search_params_json", pd.Series(dtype=str)).map(_extract_search_payload)
    working["__selectivity"] = working["__search_payload"].map(_extract_selectivity_from_payload)
    working["__p99_inflation"] = working.apply(lambda row: _row_p99_inflation(row=row, frame=working), axis=1)
    working["__p99_inflation_valid_flag"] = working["__p99_inflation"].notna().astype(int)
    working["__p99_inflation_nan_flag"] = working["__p99_inflation"].isna().astype(int)
    grouped = (
        working.groupby(["scenario", "dataset_bundle", "engine"], dropna=False, sort=True)
        .agg(
            p99_ms_median=("p99_ms", "median"),
            p99_inflation_median=("__p99_inflation", "median"),
            p99_inflation_valid_rows=("__p99_inflation_valid_flag", "sum"),
            p99_inflation_nan_rows=("__p99_inflation_nan_flag", "sum"),
            sla_violation_rate_mean=("sla_violation_rate", "mean"),
            errors_sum=("errors", "sum"),
            rows=("run_id", "count"),
        )
        .reset_index()
    )
    grouped["p99_inflation_status"] = grouped.apply(_inflation_status, axis=1)
    return grouped[columns]


def _table_t4_decision(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "dataset_bundle",
        "recommended_engine",
        "recommended_class",
        "basis",
        "median_p99_ms",
        "median_qps",
        "median_rhu_h",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"scenario", "dataset_bundle", "engine", "p99_ms", "qps", "rhu_h", "sla_threshold_ms"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    grouped = (
        frame.groupby(["scenario", "dataset_bundle", "engine"], dropna=False, sort=True)
        .agg(median_p99_ms=("p99_ms", "median"), median_qps=("qps", "median"), median_rhu_h=("rhu_h", "median"))
        .reset_index()
    )
    sla = (
        frame.groupby(["scenario", "dataset_bundle", "engine"], dropna=False, sort=True)
        .agg(sla_threshold_ms=("sla_threshold_ms", "median"))
        .reset_index()
    )
    merged = grouped.merge(sla, on=["scenario", "dataset_bundle", "engine"], how="left")

    rows: list[dict[str, Any]] = []
    for (scenario, dataset_bundle), sub in merged.groupby(["scenario", "dataset_bundle"], sort=True):
        meets = sub[sub["median_p99_ms"] <= sub["sla_threshold_ms"]]
        if not meets.empty:
            chosen = meets.sort_values(["median_rhu_h", "median_p99_ms", "median_qps"], ascending=[True, True, False]).iloc[0]
            basis = "min_rhu_h_meeting_sla"
        else:
            chosen = sub.sort_values(["median_p99_ms", "median_rhu_h", "median_qps"], ascending=[True, True, False]).iloc[0]
            basis = "min_p99_no_sla_match"
        engine = str(chosen["engine"])
        rows.append(
            {
                "scenario": scenario,
                "dataset_bundle": dataset_bundle,
                "recommended_engine": engine,
                "recommended_class": _engine_class(engine),
                "basis": basis,
                "median_p99_ms": float(chosen["median_p99_ms"]),
                "median_qps": float(chosen["median_qps"]),
                "median_rhu_h": float(chosen["median_rhu_h"]),
            }
        )
    out = pd.DataFrame(rows)
    return out[columns] if not out.empty else pd.DataFrame(columns=columns)


def _extract_selectivity(search_params_json: Any) -> float | None:
    payload = _extract_search_payload(search_params_json)
    return _extract_selectivity_from_payload(payload)


def _extract_search_payload(search_params_json: Any) -> dict[str, Any] | None:
    if not isinstance(search_params_json, str) or not search_params_json:
        return None
    try:
        payload = json.loads(search_params_json)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_selectivity_from_payload(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("selectivity")
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _row_p99_inflation(*, row: pd.Series, frame: pd.DataFrame) -> float:
    scenario = str(row.get("scenario", ""))
    engine = str(row.get("engine", ""))
    dataset = str(row.get("dataset_bundle", ""))
    dataset_hash = row.get("dataset_hash")
    payload = row.get("__search_payload")
    p99 = float(row.get("p99_ms", 0.0))
    if p99 <= 0:
        return float("nan")
    baseline = None
    if scenario == "s2_filtered_ann":
        recorded = _extract_float_from_payload(payload, "p99_inflation_vs_unfiltered")
        if recorded is not None and recorded >= 0.0:
            return recorded
        mask = (
            (frame["scenario"] == "s2_filtered_ann")
            & (frame["engine"] == engine)
            & (frame["dataset_bundle"] == dataset)
            & (frame["__selectivity"] == 1.0)
        )
        if mask.any():
            baseline = float(frame.loc[mask, "p99_ms"].median())
    elif scenario in {"s3_churn_smooth", "s3b_churn_bursty"}:
        if isinstance(payload, dict):
            missing = payload.get("s1_baseline_missing")
            if isinstance(missing, bool) and missing:
                return float("nan")
            recorded = _extract_float_from_payload(payload, "p99_inflation_vs_s1_baseline")
            if recorded is not None and recorded >= 0.0:
                return recorded
        clients_read = int(row.get("clients_read", 0))
        mask = (
            (frame["scenario"] == "s1_ann_frontier")
            & (frame["engine"] == engine)
            & (frame["dataset_bundle"] == dataset)
            & (frame.get("clients_read", pd.Series(dtype=int)) == clients_read)
        )
        if "dataset_hash" in frame.columns and dataset_hash is not None:
            mask = mask & (frame["dataset_hash"] == dataset_hash)
        if mask.any():
            baseline = float(frame.loc[mask, "p99_ms"].median())
    if baseline is None or baseline <= 0:
        return float("nan")
    return p99 / baseline


def _extract_float_from_payload(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _inflation_status(row: pd.Series) -> str:
    valid = int(row.get("p99_inflation_valid_rows", 0))
    missing = int(row.get("p99_inflation_nan_rows", 0))
    if valid <= 0 and missing > 0:
        return "not_computable"
    if valid > 0 and missing > 0:
        return "computed_partial_rows"
    return "computed_all_rows"


def _engine_class(engine: str) -> str:
    lower = engine.lower()
    if lower == "qdrant":
        return "vector-first"
    if lower in {"pgvector", "postgresql", "postgres"}:
        return "db-first"
    if lower in {"lancedb-inproc", "lancedb-service", "lancedb_inproc", "lancedb_service"}:
        return "embedded/local"
    if lower in {"faiss-cpu", "faiss_cpu"}:
        return "baseline"
    return "other"
