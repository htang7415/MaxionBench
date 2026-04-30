"""Portable-agentic report export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .plots import (
    DPI,
    FIGURE_FACE_COLOR,
    FONT_SIZE,
    PANEL_HEIGHT_IN,
    PANEL_PX,
    PANEL_WIDTH_IN,
    STYLE_VERSION,
    TEXT_COLOR,
    GRID_COLOR,
    ENGINE_PALETTE,
    load_results,
)
from maxionbench.tools.verify_engine_readiness import BEHAVIOR_CARD_BY_ADAPTER, REQUIRED_ADAPTERS

import matplotlib.pyplot as plt


_PORTABLE_SCENARIOS = {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}
_BUDGET_ORDER = {"b0": 0, "b1": 1, "b2": 2}
_BUDGET_PAIRS = [("b0", "b1"), ("b1", "b2"), ("b0", "b2")]
_MVD_P99_MAX_MS_THRESHOLD = 200.0
_MVD_SENSITIVITY_THRESHOLDS_MS: tuple[float | None, ...] = (100.0, 200.0, 500.0, None)
_BOOTSTRAP_SEED = 20260428
_BOOTSTRAP_RESAMPLES = 2000
_COST_SENSITIVITY_MULTIPLIERS: tuple[float, ...] = (0.1, 1.0, 10.0)
_REPORT_COLUMN_ALIASES = {
    "freshness_hit_at_1s": "post_insert_hit_at_10_1s",
    "freshness_hit_at_5s": "post_insert_hit_at_10_5s",
    "freshness_floor_for_budget": "post_insert_floor_for_budget",
    "freshness_hit_at_1s_mean": "post_insert_hit_at_10_1s_mean",
    "freshness_hit_at_1s_ci95_low": "post_insert_hit_at_10_1s_ci95_low",
    "freshness_hit_at_1s_ci95_high": "post_insert_hit_at_10_1s_ci95_high",
    "freshness_hit_at_5s_mean": "post_insert_hit_at_10_5s_mean",
    "freshness_hit_at_5s_ci95_low": "post_insert_hit_at_10_5s_ci95_low",
    "freshness_hit_at_5s_ci95_high": "post_insert_hit_at_10_5s_ci95_high",
    "freshness_event_count": "post_insert_event_count",
    "freshness_ci_method": "post_insert_ci_method",
}


def generate_portable_report_bundle(
    *,
    input_dir: Path,
    out_dir: Path,
    conformance_matrix_path: Path | None = None,
    behavior_dir: Path | None = None,
) -> dict[str, list[Path]]:
    frame = load_results(input_dir)
    portable = _extract_portable_frame(frame=frame)
    if portable.empty:
        raise RuntimeError(
            f"no portable-agentic results found under {input_dir}; expected scenarios {sorted(_PORTABLE_SCENARIOS)}"
        )
    resolved_conformance_matrix_path, resolved_behavior_dir = _resolve_reportability_inputs(
        conformance_matrix_path=conformance_matrix_path,
        behavior_dir=behavior_dir,
    )
    reportability = _reportability_by_adapter(
        conformance_matrix_path=resolved_conformance_matrix_path,
        behavior_dir=resolved_behavior_dir,
    )
    reportable_engines = {
        engine
        for engine, payload in reportability.items()
        if bool(payload.get("reportable"))
    }
    portable_reportable = portable.loc[portable["engine"].astype(str).isin(reportable_engines)].copy()
    if portable_reportable.empty:
        raise RuntimeError("portable report bundle requires at least one reportable engine after conformance filtering")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = _export_portable_tables(
        frame=portable_reportable,
        observed_frame=portable,
        out_dir=out_dir,
        conformance_matrix_path=resolved_conformance_matrix_path,
        behavior_dir=resolved_behavior_dir,
    )
    figures = _export_portable_figures(frame=portable_reportable, out_dir=out_dir)
    return {"figures": figures, "tables": tables}


def _extract_portable_frame(*, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["__search_payload"] = working.get("search_params_json", pd.Series(dtype=str)).map(_extract_search_payload)
    working["profile"] = working.get("__meta_profile", pd.Series(dtype=str)).astype(str)
    working["budget_level"] = _coalesced_string_column(working, "budget_level")
    budget_from_payload = working["__search_payload"].map(lambda payload: str(payload.get("budget_level") or ""))  # type: ignore[union-attr]
    working.loc[working["budget_level"] == "", "budget_level"] = budget_from_payload[working["budget_level"] == ""]
    fallback_budget = _normalized_string_series(working.get("__meta_budget_level", pd.Series(dtype=object)))
    working.loc[working["budget_level"] == "", "budget_level"] = fallback_budget[working["budget_level"] == ""]
    working["embedding_model"] = _coalesced_string_column(working, "embedding_model")
    embedding_from_payload = working["__search_payload"].map(lambda payload: str(payload.get("embedding_model") or ""))  # type: ignore[union-attr]
    working.loc[working["embedding_model"] == "", "embedding_model"] = embedding_from_payload[working["embedding_model"] == ""]
    fallback_embedding = _normalized_string_series(working.get("__meta_embedding_model", pd.Series(dtype=object)))
    working.loc[working["embedding_model"] == "", "embedding_model"] = fallback_embedding[working["embedding_model"] == ""]
    working["task_cost_est"] = _coalesced_float_column(working, "task_cost_est")
    working["embedding_dim"] = _coalesced_float_column(working, "embedding_dim")
    embedding_dim_from_meta = pd.to_numeric(working.get("__meta_embedding_dim", pd.Series(dtype=float)), errors="coerce")
    working.loc[working["embedding_dim"].isna(), "embedding_dim"] = embedding_dim_from_meta[working["embedding_dim"].isna()]
    working["c_llm_in"] = _coalesced_float_column(working, "c_llm_in")
    c_llm_in_from_meta = pd.to_numeric(working.get("__meta_c_llm_in", pd.Series(dtype=float)), errors="coerce")
    working.loc[working["c_llm_in"].isna(), "c_llm_in"] = c_llm_in_from_meta[working["c_llm_in"].isna()]
    working["primary_quality_metric"] = working["__search_payload"].map(lambda payload: str(payload.get("primary_quality_metric") or ""))  # type: ignore[union-attr]
    working["primary_quality_value"] = working["__search_payload"].map(lambda payload: _payload_float(payload, "primary_quality_value"))
    working["observation_path"] = working["__search_payload"].map(lambda payload: str(payload.get("observation_path") or ""))  # type: ignore[union-attr]
    for key in (
        "freshness_hit_at_1s",
        "freshness_hit_at_5s",
        "stale_answer_rate_at_5s",
        "p95_visibility_latency_ms",
        "evidence_coverage_at_5",
        "evidence_coverage_at_10",
        "evidence_coverage_at_20",
        "avg_retrieved_input_tokens",
        "retrieval_cost_est",
        "embedding_cost_est",
        "llm_context_cost_est",
    ):
        working[key] = _coalesced_float_column(working, key)
    for int_key in ("event_count", "overlap_skipped_event_count"):
        working[int_key] = working["__search_payload"].map(
            lambda payload, k=int_key: int(payload[k]) if isinstance(payload, dict) and k in payload else None  # type: ignore[union-attr]
        )

    mask = working["scenario"].astype(str).isin(_PORTABLE_SCENARIOS) | (working["profile"] == "portable-agentic")
    portable = working.loc[mask].copy()
    if portable.empty:
        return portable
    portable["budget_sort"] = portable["budget_level"].map(lambda value: _BUDGET_ORDER.get(str(value).lower(), 999))
    portable = portable.sort_values(
        ["scenario", "budget_sort", "engine", "embedding_model", "quality_target", "repeat_idx"],
        kind="stable",
    ).reset_index(drop=True)
    return portable


def _export_portable_tables(
    *,
    frame: pd.DataFrame,
    observed_frame: pd.DataFrame,
    out_dir: Path,
    conformance_matrix_path: Path | None,
    behavior_dir: Path | None,
) -> list[Path]:
    tables: list[Path] = []
    summary = frame[
        [
            "run_id",
            "scenario",
            "budget_level",
            "engine",
            "embedding_model",
            "quality_target",
            "primary_quality_metric",
            "primary_quality_value",
            "p99_ms",
            "qps",
            "task_cost_est",
            "freshness_hit_at_5s",
            "stale_answer_rate_at_5s",
            "evidence_coverage_at_10",
            "event_count",
            "overlap_skipped_event_count",
        ]
    ].copy()
    for int_col in ("event_count", "overlap_skipped_event_count"):
        summary[int_col] = pd.to_numeric(summary[int_col], errors="coerce").astype("Int64")
    summary = summary.rename(columns=_REPORT_COLUMN_ALIASES)
    summary_path = out_dir / "portable_summary.csv"
    summary.to_csv(summary_path, index=False)
    tables.append(summary_path)

    winners = _winner_rows(frame=frame)
    winners_path = out_dir / "portable_winners.csv"
    winners_csv = _csv_safe_frame(winners).rename(columns=_REPORT_COLUMN_ALIASES)
    winners_csv = winners_csv.drop(
        columns=[col for col in winners_csv.columns if col.startswith("__") or col == "search_params_json"],
        errors="ignore",
    )
    winners_csv.to_csv(winners_path, index=False)
    tables.append(winners_path)

    stability = _stability_table(winners=winners)
    stability_path = out_dir / "portable_stability.csv"
    stability.to_csv(stability_path, index=False)
    tables.append(stability_path)

    deployment = _minimum_viable_deployment_table(winners=winners)
    deployment_path = out_dir / "minimum_viable_deployment.csv"
    deployment.to_csv(deployment_path, index=False)
    tables.append(deployment_path)

    deployment_sensitivity = _minimum_viable_deployment_sensitivity_table(winners=winners)
    deployment_sensitivity_path = out_dir / "minimum_viable_deployment_sensitivity.csv"
    deployment_sensitivity.to_csv(deployment_sensitivity_path, index=False)
    tables.append(deployment_sensitivity_path)

    decision = _portable_decision_table(winners=winners, stability=stability)
    decision_path = out_dir / "portable_decision_table.csv"
    decision.to_csv(decision_path, index=False)
    tables.append(decision_path)
    decision_tex_path = out_dir / "portable_decision_table.tex"
    decision_tex_path.write_text(_portable_decision_table_latex(table=decision), encoding="utf-8")
    tables.append(decision_tex_path)

    neurips_main = _neurips_main_results_table(frame=frame, winners=winners, stability=stability)
    neurips_main_path = out_dir / "neurips_main_results.csv"
    neurips_main.to_csv(neurips_main_path, index=False)
    tables.append(neurips_main_path)
    neurips_main_tex_path = out_dir / "neurips_main_results.tex"
    neurips_main_tex_path.write_text(_neurips_main_results_latex(table=neurips_main), encoding="utf-8")
    tables.append(neurips_main_tex_path)

    decision_error = _decision_error_ablation_table(decision=decision, stability=stability)
    decision_error_path = out_dir / "decision_error_ablation.csv"
    decision_error.to_csv(decision_error_path, index=False)
    tables.append(decision_error_path)
    decision_error_tex_path = out_dir / "decision_error_ablation.tex"
    decision_error_tex_path.write_text(_decision_error_ablation_latex(table=decision_error), encoding="utf-8")
    tables.append(decision_error_tex_path)

    cost_formula = _cost_formula_table()
    cost_formula_path = out_dir / "cost_formula.csv"
    cost_formula.to_csv(cost_formula_path, index=False)
    tables.append(cost_formula_path)
    cost_formula_tex_path = out_dir / "cost_formula.tex"
    cost_formula_tex_path.write_text(_cost_formula_latex(table=cost_formula), encoding="utf-8")
    tables.append(cost_formula_tex_path)

    cost_sensitivity = _cost_sensitivity_table(winners=winners)
    cost_sensitivity_path = out_dir / "cost_sensitivity.csv"
    cost_sensitivity.to_csv(cost_sensitivity_path, index=False)
    tables.append(cost_sensitivity_path)
    cost_sensitivity_tex_path = out_dir / "cost_sensitivity.tex"
    cost_sensitivity_tex_path.write_text(_cost_sensitivity_latex(table=cost_sensitivity), encoding="utf-8")
    tables.append(cost_sensitivity_tex_path)

    latency_distribution = _latency_distribution_table(winners=winners, decision=decision)
    latency_distribution_path = out_dir / "latency_distribution.csv"
    latency_distribution.to_csv(latency_distribution_path, index=False)
    tables.append(latency_distribution_path)
    latency_distribution_tex_path = out_dir / "latency_distribution.tex"
    latency_distribution_tex_path.write_text(_latency_distribution_latex(table=latency_distribution), encoding="utf-8")
    tables.append(latency_distribution_tex_path)

    support = _support_table(
        frame=observed_frame,
        winners=winners,
        conformance_matrix_path=conformance_matrix_path,
        behavior_dir=behavior_dir,
    )
    support_path = out_dir / "portable_support_table.csv"
    support.to_csv(support_path, index=False)
    tables.append(support_path)
    support_tex_path = out_dir / "portable_support_table.tex"
    support_tex_path.write_text(_support_table_latex(table=support), encoding="utf-8")
    tables.append(support_tex_path)

    engine_configuration = _engine_configuration_table(frame=frame, support=support)
    engine_configuration_path = out_dir / "engine_configuration.csv"
    engine_configuration.to_csv(engine_configuration_path, index=False)
    tables.append(engine_configuration_path)
    engine_configuration_tex_path = out_dir / "engine_configuration.tex"
    engine_configuration_tex_path.write_text(_engine_configuration_latex(table=engine_configuration), encoding="utf-8")
    tables.append(engine_configuration_tex_path)

    s3_all_evidence = _s3_all_evidence_hit_table(winners=winners, decision=decision)
    s3_all_evidence_path = out_dir / "s3_all_evidence_hit.csv"
    s3_all_evidence.to_csv(s3_all_evidence_path, index=False)
    tables.append(s3_all_evidence_path)
    s3_all_evidence_tex_path = out_dir / "s3_all_evidence_hit.tex"
    s3_all_evidence_tex_path.write_text(_s3_all_evidence_hit_latex(table=s3_all_evidence), encoding="utf-8")
    tables.append(s3_all_evidence_tex_path)

    meta_path = out_dir / "portable_summary.meta.json"
    meta_payload = {
        "mode": "portable-agentic",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "rows_total": int(len(frame)),
        "observed_rows_total": int(len(observed_frame)),
        "winner_rows": int(len(winners)),
        "table_names": [path.name for path in tables],
        "budgets": sorted({str(value) for value in frame["budget_level"].tolist() if str(value)}),
        "scenarios": sorted({str(value) for value in frame["scenario"].tolist() if str(value)}),
        "engines": sorted({str(value) for value in frame["engine"].tolist() if str(value)}),
        "support_table_rows": int(len(support)),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tables.append(meta_path)
    return tables


def _export_portable_figures(*, frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    figures: list[Path] = []
    _set_plot_style()
    winners = _winner_rows(frame=frame)
    stability = _stability_table(winners=winners)

    task_cost_path = out_dir / "portable_task_cost_by_budget.png"
    fig, ax = _paper_figure()
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_task_cost_by_budget(ax=ax, winners=winners)
    _save_paper_figure(fig=fig, path=task_cost_path)
    plt.close(fig)
    _write_meta(
        task_cost_path,
        {
            "figure_name": "portable_task_cost_by_budget",
            "mode": "portable-agentic",
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "rows_used": int(len(winners)),
            "scenarios": sorted({str(value) for value in winners["scenario"].tolist()}),
            "budgets": sorted({str(value) for value in winners["budget_level"].tolist()}),
        },
    )
    figures.append(task_cost_path)

    stability_path = out_dir / "portable_budget_stability.png"
    fig, ax = _paper_figure()
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_budget_stability(ax=ax, stability=stability)
    _save_paper_figure(fig=fig, path=stability_path)
    plt.close(fig)
    _write_meta(
        stability_path,
        {
            "figure_name": "portable_budget_stability",
            "mode": "portable-agentic",
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "rows_used": int(len(stability)),
            "scenario_budget_pairs": stability[["scenario", "budget_pair"]].astype(str).to_dict(orient="records"),
        },
    )
    figures.append(stability_path)

    post_insert_path = out_dir / "portable_s2_post_insert_retrievability.png"
    fig, ax = _paper_figure()
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_s2_post_insert_retrievability(ax=ax, winners=winners)
    _save_paper_figure(fig=fig, path=post_insert_path)
    plt.close(fig)
    _write_meta(
        post_insert_path,
        {
            "figure_name": "portable_s2_post_insert_retrievability",
            "mode": "portable-agentic",
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "rows_used": int(len(winners.loc[winners["scenario"] == "s2_streaming_memory"])),
        },
    )
    figures.append(post_insert_path)

    mvd_sensitivity = _minimum_viable_deployment_sensitivity_table(winners=winners)
    mvd_sensitivity_path = out_dir / "portable_mvd_sensitivity.png"
    fig, ax = _paper_figure(height_in=3.0)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_mvd_sensitivity(ax=ax, sensitivity=mvd_sensitivity)
    _save_paper_figure(fig=fig, path=mvd_sensitivity_path)
    plt.close(fig)
    _write_meta(
        mvd_sensitivity_path,
        {
            "figure_name": "portable_mvd_sensitivity",
            "mode": "portable-agentic",
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "rows_used": int(len(mvd_sensitivity)),
            "p99_thresholds": [str(value) for value in mvd_sensitivity["p99_max_threshold_ms"].tolist()],
        },
    )
    figures.append(mvd_sensitivity_path)
    return figures


def _winner_rows(*, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working = working.sort_values(
        ["scenario", "budget_sort", "clients_read", "engine", "embedding_model", "task_cost_est", "p99_ms", "qps"],
        ascending=[True, True, True, True, True, True, True, False],
        kind="stable",
    )
    grouped = (
        working.groupby(["scenario", "budget_level", "clients_read", "engine", "embedding_model"], dropna=False, as_index=False)
        .first()
        .reset_index(drop=True)
    )
    grouped["rank_within_budget"] = grouped.groupby(["scenario", "budget_level", "clients_read"], dropna=False)["task_cost_est"].rank(
        method="dense",
        ascending=True,
    )
    return grouped.sort_values(["scenario", "budget_sort", "clients_read", "rank_within_budget", "engine"], kind="stable").reset_index(drop=True)


def _stability_table(*, winners: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in winners.groupby("scenario", dropna=False):
        scenario_rows = scenario_frame.copy()
        key_cols = ["clients_read", "engine", "embedding_model"]
        for left_budget, right_budget in _BUDGET_PAIRS:
            left = scenario_rows.loc[scenario_rows["budget_level"] == left_budget, key_cols + ["task_cost_est", "rank_within_budget"]]
            right = scenario_rows.loc[scenario_rows["budget_level"] == right_budget, key_cols + ["task_cost_est", "rank_within_budget"]]
            if left.empty or right.empty:
                continue
            merged = left.merge(right, on=key_cols, suffixes=("_left", "_right"))
            if merged.empty:
                continue
            rho = _spearman_rank_correlation(
                merged["rank_within_budget_left"].tolist(),
                merged["rank_within_budget_right"].tolist(),
            )
            left_top1 = set(
                merged.loc[merged["rank_within_budget_left"] == merged["rank_within_budget_left"].min(), "engine"].tolist()
            )
            right_top1 = set(
                merged.loc[merged["rank_within_budget_right"] == merged["rank_within_budget_right"].min(), "engine"].tolist()
            )
            left_top2 = set(
                merged.loc[merged["rank_within_budget_left"] <= 2, "engine"].tolist()
            )
            right_top2 = set(
                merged.loc[merged["rank_within_budget_right"] <= 2, "engine"].tolist()
            )
            rows.append(
                {
                    "scenario": str(scenario),
                    "budget_pair": f"{left_budget}->{right_budget}",
                    "spearman_rho": rho,
                    "top1_agreement": float(bool(left_top1 & right_top1)),
                    "top2_agreement": float(bool(left_top2 & right_top2)),
                    "common_engine_embedding_pairs": int(len(merged)),
                    "clients_read_values": ",".join(sorted({str(value) for value in merged["clients_read"].tolist()})),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "budget_pair"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["scenario", "budget_pair", "spearman_rho", "top1_agreement", "top2_agreement", "common_engine_embedding_pairs", "clients_read_values"]
    )


def _minimum_viable_deployment_table(
    *,
    winners: pd.DataFrame,
    p99_max_threshold_ms: float | None = _MVD_P99_MAX_MS_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in winners.groupby("scenario", dropna=False):
        preferred = scenario_frame.loc[scenario_frame["budget_level"] == "b2"]
        if preferred.empty:
            preferred = scenario_frame.sort_values(["budget_sort", "rank_within_budget"], kind="stable").tail(1)
        quality_floor = _quality_floor(str(scenario))

        # Aggregate across concurrency levels: mean cost, mean quality, max p99 (worst-case latency)
        agg_cols: dict[str, Any] = {
            "primary_quality_value": ("primary_quality_value", "mean"),
            "task_cost_est": ("task_cost_est", "mean"),
            "p99_ms_mean": ("p99_ms", "mean"),
            "p99_ms_max": ("p99_ms", "max"),
            "qps": ("qps", "mean"),
            "primary_quality_metric": ("primary_quality_metric", "first"),
            "embedding_model": ("embedding_model", "first"),
        }
        if "freshness_hit_at_5s" in preferred.columns:
            agg_cols["freshness_hit_at_5s"] = ("freshness_hit_at_5s", "mean")
        if "errors" in preferred.columns:
            agg_cols["errors"] = ("errors", "sum")
        aggregated = (
            preferred.groupby(["engine", "embedding_model"], dropna=False, as_index=False)
            .agg(**agg_cols)
        )

        eligible = aggregated.loc[pd.to_numeric(aggregated["primary_quality_value"], errors="coerce") >= quality_floor]
        if "errors" in eligible.columns:
            eligible = eligible.loc[pd.to_numeric(eligible["errors"], errors="coerce").fillna(0.0) <= 0.0]
        if eligible.empty:
            eligible = aggregated

        # Prefer engines whose worst-case p99 (across concurrency) stays below the deployment SLA.
        # This prevents cost-optimal but concurrency-hostile engines from appearing
        # as the minimum viable deployment recommendation.
        if p99_max_threshold_ms is not None:
            eligible_fast = eligible.loc[pd.to_numeric(eligible["p99_ms_max"], errors="coerce") <= p99_max_threshold_ms]
            if not eligible_fast.empty:
                eligible = eligible_fast

        # Primary sort: lowest mean task cost; tie-break by worst-case p99, then best mean qps
        best = eligible.sort_values(
            ["task_cost_est", "p99_ms_max", "qps"],
            ascending=[True, True, False],
            kind="stable",
        ).iloc[0]

        reason_parts = [
            f"{best['primary_quality_metric']}={float(best['primary_quality_value']):.3f}",
        ]
        freshness_val = pd.to_numeric(pd.Series([best.get("freshness_hit_at_5s")]), errors="coerce").iloc[0]
        if not math.isnan(freshness_val):
            reason_parts.append(f"post_insert_hit@10,5s={float(freshness_val):.3f}")
        reason_parts.append(f"p99_mean={float(best['p99_ms_mean']):.1f}ms")
        reason_parts.append(f"p99_max={float(best['p99_ms_max']):.1f}ms")
        reason_parts.append(f"task_cost={float(best['task_cost_est']):.6f}")
        reason = ", ".join(reason_parts)
        rows.append(
            {
                "workload_type": str(scenario),
                "minimum_engine": str(best["engine"]),
                "recommended_embedding_tier": str(best["embedding_model"]),
                "why": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("workload_type", kind="stable").reset_index(drop=True)


def _minimum_viable_deployment_sensitivity_table(*, winners: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for threshold in _MVD_SENSITIVITY_THRESHOLDS_MS:
        deployment = _minimum_viable_deployment_table(winners=winners, p99_max_threshold_ms=threshold).copy()
        deployment.insert(0, "p99_max_threshold_ms", threshold if threshold is not None else "none")
        frames.append(deployment)
    if not frames:
        return pd.DataFrame(
            columns=[
                "p99_max_threshold_ms",
                "workload_type",
                "minimum_engine",
                "recommended_embedding_tier",
                "why",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["workload_type", "p99_max_threshold_ms"],
        key=lambda series: series.map(lambda value: float("inf") if value == "none" else value) if series.name == "p99_max_threshold_ms" else series,
        kind="stable",
    ).reset_index(drop=True)


def _portable_decision_table(*, winners: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    strict = _minimum_viable_deployment_table(winners=winners, p99_max_threshold_ms=_MVD_P99_MAX_MS_THRESHOLD)
    unconstrained = _minimum_viable_deployment_table(winners=winners, p99_max_threshold_ms=None)
    rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in winners.groupby("scenario", dropna=False):
        scenario_name = str(scenario)
        preferred = scenario_frame.loc[scenario_frame["budget_level"].astype(str) == "b2"].copy()
        if preferred.empty:
            preferred = scenario_frame.sort_values(["budget_sort", "rank_within_budget"], kind="stable").tail(1)
        quality_floor = _quality_floor(scenario_name)
        aggregated = _aggregate_decision_candidates(preferred=preferred)
        eligible_quality = aggregated.loc[pd.to_numeric(aggregated["primary_quality_value"], errors="coerce") >= quality_floor]
        if eligible_quality.empty:
            eligible_quality = aggregated
        quality_best = eligible_quality.sort_values(
            ["primary_quality_value", "p99_ms_max", "task_cost_est"],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]
        strict_row = strict.loc[strict["workload_type"].astype(str) == scenario_name].iloc[0]
        unconstrained_row = unconstrained.loc[unconstrained["workload_type"].astype(str) == scenario_name].iloc[0]
        stability_fields = _stability_fields(stability=stability, scenario=scenario_name)
        rows.append(
            {
                "scenario": scenario_name,
                "strict_p99_threshold_ms": _MVD_P99_MAX_MS_THRESHOLD,
                "strict_p99_engine": str(strict_row["minimum_engine"]),
                "strict_p99_embedding_model": str(strict_row["recommended_embedding_tier"]),
                "unconstrained_cost_engine": str(unconstrained_row["minimum_engine"]),
                "unconstrained_cost_embedding_model": str(unconstrained_row["recommended_embedding_tier"]),
                "quality_winner_engine": str(quality_best["engine"]),
                "quality_winner_embedding_model": str(quality_best["embedding_model"]),
                "quality_winner_metric": str(quality_best["primary_quality_metric"]),
                "quality_winner_value": float(quality_best["primary_quality_value"]),
                "quality_winner_p99_ms_max": float(quality_best["p99_ms_max"]),
                "quality_winner_task_cost_est": float(quality_best["task_cost_est"]),
                "spearman_b0_b2": stability_fields["spearman_b0_b2"],
                "top1_agreement_b0_b2": stability_fields["top1_agreement_b0_b2"],
                "top2_agreement_b0_b2": stability_fields["top2_agreement_b0_b2"],
                "decision_stability_note": stability_fields["decision_stability_note"],
            }
        )
    columns = [
        "scenario",
        "strict_p99_threshold_ms",
        "strict_p99_engine",
        "strict_p99_embedding_model",
        "unconstrained_cost_engine",
        "unconstrained_cost_embedding_model",
        "quality_winner_engine",
        "quality_winner_embedding_model",
        "quality_winner_metric",
        "quality_winner_value",
        "quality_winner_p99_ms_max",
        "quality_winner_task_cost_est",
        "spearman_b0_b2",
        "top1_agreement_b0_b2",
        "top2_agreement_b0_b2",
        "decision_stability_note",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("scenario", kind="stable").reset_index(drop=True)


def _aggregate_decision_candidates(*, preferred: pd.DataFrame) -> pd.DataFrame:
    agg_cols: dict[str, Any] = {
        "primary_quality_value": ("primary_quality_value", "mean"),
        "task_cost_est": ("task_cost_est", "mean"),
        "p99_ms_max": ("p99_ms", "max"),
        "qps": ("qps", "mean"),
        "primary_quality_metric": ("primary_quality_metric", "first"),
    }
    return preferred.groupby(["engine", "embedding_model"], dropna=False, as_index=False).agg(**agg_cols)


def _decision_error_ablation_table(*, decision: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    s3 = decision.loc[decision["scenario"].astype(str) == "s3_multi_hop"]
    s3_shift = "S3 no-p99 cost choice shifts to LanceDB/bge-small instead of FAISS/bge-small"
    if not s3.empty:
        row = s3.iloc[0]
        s3_shift = (
            f"S3 no-p99 cost choice shifts to {row['unconstrained_cost_engine']}/"
            f"{_short_embedding_label(str(row['unconstrained_cost_embedding_model']))} instead of "
            f"{row['strict_p99_engine']}/{_short_embedding_label(str(row['strict_p99_embedding_model']))}"
        )
    changed_quality = []
    for _, row in decision.iterrows():
        strict = (str(row["strict_p99_engine"]), str(row["strict_p99_embedding_model"]))
        quality = (str(row["quality_winner_engine"]), str(row["quality_winner_embedding_model"]))
        if strict != quality:
            changed_quality.append(_short_scenario_code(str(row["scenario"])))
    unstable = []
    if not stability.empty:
        b0_b2 = stability.loc[stability["budget_pair"].astype(str) == "b0->b2"]
        for _, row in b0_b2.iterrows():
            if _safe_float(row.get("top1_agreement")) < 1.0:
                unstable.append(_short_scenario_code(str(row["scenario"])))

    rows = [
        {
            "missing_protocol_component": "p99 latency gate",
            "wrong_conclusion_caused_by_omission": s3_shift,
            "manuscript_evidence": "Table 2 and p99 sensitivity table",
            "source_path": "portable_decision_table.csv; minimum_viable_deployment_sensitivity.csv",
        },
        {
            "missing_protocol_component": "Objective separation",
            "wrong_conclusion_caused_by_omission": (
                "Quality-first decisions differ from strict-latency decisions on "
                f"{', '.join(changed_quality) if changed_quality else 'no workloads'}"
            ),
            "manuscript_evidence": "Table 2",
            "source_path": "portable_decision_table.csv",
        },
        {
            "missing_protocol_component": "Budget ladder",
            "wrong_conclusion_caused_by_omission": (
                "B0 screening fails to preserve B2 top-1 decisions for "
                f"{', '.join(unstable) if unstable else 'no workloads'}"
            ),
            "manuscript_evidence": "Figure 1 and Table 2",
            "source_path": "portable_stability.csv; portable_mvd_sensitivity.meta.json",
        },
        {
            "missing_protocol_component": "Paired S3 audit",
            "wrong_conclusion_caused_by_omission": (
                "pgvector/bge-base appears quality-first, but the matched 5,000-query audit removes the substantive margin"
            ),
            "manuscript_evidence": "Table 3",
            "source_path": "paper/experiments/s3_paired_quality/summary.json",
        },
        {
            "missing_protocol_component": "Paired S2 competitor audit",
            "wrong_conclusion_caused_by_omission": "Qdrant does not hide a quality or post-insert retrievability win",
            "manuscript_evidence": "Table 5",
            "source_path": "paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json",
        },
        {
            "missing_protocol_component": "Conformance gate",
            "wrong_conclusion_caused_by_omission": (
                "Engines without passing local conformance rows do not enter paper-facing result tables"
            ),
            "manuscript_evidence": "Section 2.1 and artifact audit",
            "source_path": "artifacts/conformance/conformance_matrix.csv; docs/behavior",
        },
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "missing_protocol_component",
            "wrong_conclusion_caused_by_omission",
            "manuscript_evidence",
            "source_path",
        ],
    )


def _cost_formula_table() -> pd.DataFrame:
    rows = [
        {
            "term": "C_retrieval",
            "meaning": "Per-query retrieval cost estimate",
            "unit": "project-defined normalized cost",
            "value_source": "run result search_params_json or results.parquet",
            "source_path": "artifacts/runs/portable/**/results.parquet",
        },
        {
            "term": "C_embedding",
            "meaning": "Per-query embedding cost estimate",
            "unit": "project-defined normalized cost",
            "value_source": "embedding tier cost config recorded in run result payload",
            "source_path": "artifacts/runs/portable/**/results.parquet",
        },
        {
            "term": "c_llm_in",
            "meaning": "Input-token cost coefficient",
            "unit": "project-defined normalized cost per token",
            "value_source": "resolved run config and run metadata",
            "source_path": "artifacts/runs/portable/**/config_resolved.yaml",
        },
        {
            "term": "Nretrieved_input_tokens",
            "meaning": "Retrieved context tokens packed downstream",
            "unit": "tokens/query",
            "value_source": "measured top-k output token counts",
            "source_path": "artifacts/runs/portable/**/results.parquet",
        },
        {
            "term": "task_cost_est",
            "meaning": "C_retrieval + C_embedding + c_llm_in x Nretrieved_input_tokens",
            "unit": "normalized task cost/query",
            "value_source": "computed report table",
            "source_path": "portable_winners.csv",
        },
    ]
    return pd.DataFrame(rows, columns=["term", "meaning", "unit", "value_source", "source_path"])


def _cost_sensitivity_table(*, winners: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, candidates in _strict_candidate_groups(winners=winners).items():
        if candidates.empty:
            continue
        displayed = candidates.head(2).copy()
        base_c_llm = _candidate_c_llm_in(candidates.iloc[0])
        if not math.isfinite(base_c_llm) or base_c_llm == 0.0:
            base_c_llm = 0.15
        for multiplier in _COST_SENSITIVITY_MULTIPLIERS:
            scored = displayed.copy()
            scored["sensitivity_task_cost_est"] = scored.apply(
                lambda row: _sensitivity_task_cost(row=row, multiplier=multiplier, base_c_llm=base_c_llm),
                axis=1,
            )
            selected = scored.sort_values(
                ["sensitivity_task_cost_est", "p99_ms_max", "qps"],
                ascending=[True, True, False],
                kind="stable",
            ).iloc[0]
            selected_config = _candidate_label(selected)
            main_config = _candidate_label(candidates.iloc[0])
            for _, candidate in scored.iterrows():
                rows.append(
                    {
                        "workload": str(scenario),
                        "candidate_role": "strict choice" if _candidate_label(candidate) == main_config else "nearest strict-cost competitor",
                        "engine": str(candidate["engine"]),
                        "embedding_model": str(candidate["embedding_model"]),
                        "c_llm_in_multiplier": multiplier,
                        "retrieval_cost_est": _safe_float(candidate.get("retrieval_cost_est")),
                        "embedding_cost_est": _safe_float(candidate.get("embedding_cost_est")),
                        "avg_retrieved_input_tokens": _candidate_tokens(row=candidate, base_c_llm=base_c_llm),
                        "sensitivity_task_cost_est": _safe_float(candidate.get("sensitivity_task_cost_est")),
                        "selected_config_at_multiplier": selected_config,
                        "selection_changes_from_main": selected_config != main_config,
                        "source_path": str(candidate.get("source_path") or ""),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "workload",
            "candidate_role",
            "engine",
            "embedding_model",
            "c_llm_in_multiplier",
            "retrieval_cost_est",
            "embedding_cost_est",
            "avg_retrieved_input_tokens",
            "sensitivity_task_cost_est",
            "selected_config_at_multiplier",
            "selection_changes_from_main",
            "source_path",
        ],
    )


def _latency_distribution_table(*, winners: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in _decision_latency_targets(decision=decision, winners=winners):
        selected = winners.loc[
            (winners["scenario"].astype(str) == target["scenario"])
            & (winners["budget_level"].astype(str) == "b2")
            & (winners["engine"].astype(str) == target["engine"])
            & (winners["embedding_model"].astype(str) == target["embedding_model"])
        ].copy()
        if selected.empty:
            continue
        p50_values = pd.to_numeric(selected.get("p50_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        p95_values = pd.to_numeric(selected.get("p95_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        p99_values = pd.to_numeric(selected.get("p99_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        measure_requests = pd.to_numeric(selected.get("measure_requests", pd.Series(dtype=float)), errors="coerce").dropna()
        clients = sorted(
            {
                f"{int(_safe_float(row.get('clients_read')))} / {int(_safe_float(row.get('clients_write')))}"
                for _, row in selected.iterrows()
                if math.isfinite(_safe_float(row.get("clients_read"))) and math.isfinite(_safe_float(row.get("clients_write")))
            }
        )
        rows.append(
            {
                "workload": str(target["scenario"]),
                "row_role": str(target["role"]),
                "engine": str(target["engine"]),
                "embedding_model": str(target["embedding_model"]),
                "clients_read_write": ", ".join(clients),
                "p50_ms": float(p50_values.mean()) if not p50_values.empty else float("nan"),
                "p95_ms": float(p95_values.mean()) if not p95_values.empty else float("nan"),
                "p99_ms": float(p99_values.mean()) if not p99_values.empty else float("nan"),
                "p99_max_ms": float(p99_values.max()) if not p99_values.empty else float("nan"),
                "latency_observations": int(measure_requests.sum()) if not measure_requests.empty else int(len(selected)),
                "boundary": _latency_boundary(str(target["scenario"])),
                "source_path": _source_paths_for_frame(selected),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "workload",
            "row_role",
            "engine",
            "embedding_model",
            "clients_read_write",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "p99_max_ms",
            "latency_observations",
            "boundary",
            "source_path",
        ],
    )


def _engine_configuration_table(*, frame: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    support_by_engine = {
        str(row["engine"]): row
        for _, row in support.iterrows()
    } if support is not None and not support.empty else {}
    observed = frame.copy()
    rows: list[dict[str, Any]] = []
    for engine in sorted(set(REQUIRED_ADAPTERS) | {str(value) for value in observed.get("engine", pd.Series(dtype=object)).tolist()}):
        engine_rows = observed.loc[observed["engine"].astype(str) == engine].copy()
        config_payload = _first_config_payload(engine_rows)
        metadata_payload = _first_metadata_payload(engine_rows)
        support_row = support_by_engine.get(engine)
        behavior_card = str(support_row.get("behavior_card") if support_row is not None else BEHAVIOR_CARD_BY_ADAPTER.get(engine, ""))  # type: ignore[union-attr]
        included = bool(support_row.get("included_in_report")) if support_row is not None else False  # type: ignore[union-attr]
        dims = sorted(
            {
                str(int(value))
                for value in pd.to_numeric(engine_rows.get("embedding_dim", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            }
        )
        version = str(metadata_payload.get("engine_version") or config_payload.get("engine_version") or "not run")
        rows.append(
            {
                "engine": engine,
                "mode": _engine_mode(engine),
                "version": version,
                "index_search_configuration": _index_search_configuration(config_payload),
                "distance_metric": str(config_payload.get("metric") or "ip"),
                "embedding_dimension": ", ".join(dims) if dims else "",
                "process_model": _engine_process_model(engine),
                "flush_commit_path": _engine_flush_path(engine),
                "included_in_reported_matrix": included,
                "source_path": _engine_config_source_path(engine_rows=engine_rows, behavior_card=behavior_card),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "engine",
            "mode",
            "version",
            "index_search_configuration",
            "distance_metric",
            "embedding_dimension",
            "process_model",
            "flush_commit_path",
            "included_in_reported_matrix",
            "source_path",
        ],
    )


def _s3_all_evidence_hit_table(*, winners: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = _s3_all_evidence_targets(winners=winners, decision=decision)
    supplemental = _load_s3_all_evidence_supplement()
    for target in targets:
        selected = winners.loc[
            (winners["scenario"].astype(str) == "s3_multi_hop")
            & (winners["budget_level"].astype(str) == "b2")
            & (winners["engine"].astype(str) == target["engine"])
            & (winners["embedding_model"].astype(str) == target["embedding_model"])
        ].copy()
        observations = [
            row
            for row in _load_observations_from_frame(selected=selected, include_run_dir_glob=True)
            if str(row.get("observation_type") or "") == "quality"
        ]
        coverage_values = np.asarray(
            [
                _safe_float(row.get("evidence_coverage_at_10"))
                for row in observations
                if math.isfinite(_safe_float(row.get("evidence_coverage_at_10")))
            ],
            dtype=np.float64,
        )
        aggregate_values = pd.to_numeric(selected.get("evidence_coverage_at_10", pd.Series(dtype=float)), errors="coerce").dropna()
        if coverage_values.size:
            all_hit_values = np.asarray([1.0 if value >= 1.0 - 1e-12 else 0.0 for value in coverage_values], dtype=np.float64)
            all_hit = float(np.mean(all_hit_values))
            coverage = float(np.mean(coverage_values))
            method = "query-level observations; all_evidence_hit@10 is 1[evidence_coverage_at_10 == 1]"
            samples = int(coverage_values.size)
        else:
            all_hit = float("nan")
            coverage = float(aggregate_values.mean()) if not aggregate_values.empty else float("nan")
            method = "query-level observations unavailable; aggregate evidence_coverage@10 retained"
            samples = 0
        supplement = supplemental.get((str(target["engine"]), str(target["embedding_model"])))
        row_role = str(target["role"])
        source_path = _source_paths_for_frame(selected)
        if supplement is not None and _selected_from_archived_portable_run(selected):
            row_role = str(supplement.get("role") or row_role)
            coverage = _safe_float(supplement.get("evidence_coverage_at_10"))
            all_hit = _safe_float(supplement.get("all_evidence_hit_at_10"))
            samples = int(_safe_float(supplement.get("query_level_observations")))
            method = "query-level S3 all-evidence audit summary"
            source_paths = supplement.get("source_paths") or []
            if isinstance(source_paths, list):
                source_path = "; ".join(str(item) for item in source_paths[:3])
                if len(source_paths) > 3:
                    source_path = f"{source_path}; ..."
        rows.append(
            {
                "row_role": row_role,
                "engine": str(target["engine"]),
                "embedding_model": str(target["embedding_model"]),
                "evidence_coverage_at_10": coverage,
                "all_evidence_hit_at_10": all_hit,
                "query_level_observations": samples,
                "method": method,
                "source_path": source_path,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "row_role",
            "engine",
            "embedding_model",
            "evidence_coverage_at_10",
            "all_evidence_hit_at_10",
            "query_level_observations",
            "method",
            "source_path",
        ],
    )


def _selected_from_archived_portable_run(selected: pd.DataFrame) -> bool:
    for col in ("__run_path", "observation_path"):
        if col not in selected.columns:
            continue
        for raw in selected[col].dropna().astype(str).tolist():
            if "artifacts/runs/portable/" in raw.replace("\\", "/"):
                return True
    return False


def _load_s3_all_evidence_supplement() -> dict[tuple[str, str], dict[str, Any]]:
    path = Path("paper/experiments/s3_all_evidence/summary.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine") or "")
        embedding = str(row.get("embedding_model") or "")
        if engine and embedding:
            by_key[(engine, embedding)] = dict(row)
    return by_key


def _strict_candidate_groups(*, winners: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for scenario, scenario_frame in winners.groupby("scenario", dropna=False):
        preferred = scenario_frame.loc[scenario_frame["budget_level"].astype(str) == "b2"].copy()
        if preferred.empty:
            continue
        agg_spec: dict[str, Any] = {
            "primary_quality_value": ("primary_quality_value", "mean"),
            "primary_quality_metric": ("primary_quality_metric", "first"),
            "task_cost_est": ("task_cost_est", "mean"),
            "p99_ms_max": ("p99_ms", "max"),
            "qps": ("qps", "mean"),
        }
        for col in (
            "retrieval_cost_est",
            "embedding_cost_est",
            "llm_context_cost_est",
            "avg_retrieved_input_tokens",
            "c_llm_in",
        ):
            if col in preferred.columns:
                agg_spec[col] = (col, "mean")
        aggregated = preferred.groupby(["engine", "embedding_model"], dropna=False, as_index=False).agg(**agg_spec)
        source_paths = []
        for _, row in aggregated.iterrows():
            selected = preferred.loc[
                (preferred["engine"].astype(str) == str(row["engine"]))
                & (preferred["embedding_model"].astype(str) == str(row["embedding_model"]))
            ]
            source_paths.append(_source_paths_for_frame(selected))
        aggregated["source_path"] = source_paths
        eligible = aggregated.loc[pd.to_numeric(aggregated["primary_quality_value"], errors="coerce") >= _quality_floor(str(scenario))]
        if eligible.empty:
            eligible = aggregated
        fast = eligible.loc[pd.to_numeric(eligible["p99_ms_max"], errors="coerce") <= _MVD_P99_MAX_MS_THRESHOLD]
        if not fast.empty:
            eligible = fast
        grouped[str(scenario)] = eligible.sort_values(
            ["task_cost_est", "p99_ms_max", "qps"],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)
    return grouped


def _candidate_c_llm_in(row: pd.Series) -> float:
    configured = _safe_float(row.get("c_llm_in"))
    if math.isfinite(configured) and configured != 0.0:
        return configured
    context_cost = _safe_float(row.get("llm_context_cost_est"))
    tokens = _safe_float(row.get("avg_retrieved_input_tokens"))
    if math.isfinite(context_cost) and math.isfinite(tokens) and tokens != 0.0:
        return context_cost / tokens
    return float("nan")


def _sensitivity_task_cost(*, row: pd.Series, multiplier: float, base_c_llm: float) -> float:
    retrieval = _safe_float(row.get("retrieval_cost_est"))
    embedding = _safe_float(row.get("embedding_cost_est"))
    if not math.isfinite(retrieval):
        retrieval = 0.0
    if not math.isfinite(embedding):
        embedding = 0.0
    tokens = _candidate_tokens(row=row, base_c_llm=base_c_llm)
    if not math.isfinite(tokens):
        archived_cost = _safe_float(row.get("task_cost_est"))
        return archived_cost if math.isfinite(archived_cost) else float("nan")
    return retrieval + embedding + (base_c_llm * multiplier * tokens)


def _candidate_tokens(*, row: pd.Series, base_c_llm: float) -> float:
    tokens = _safe_float(row.get("avg_retrieved_input_tokens"))
    if math.isfinite(tokens):
        return tokens
    archived_cost = _safe_float(row.get("task_cost_est"))
    retrieval = _safe_float(row.get("retrieval_cost_est"))
    embedding = _safe_float(row.get("embedding_cost_est"))
    retrieval = retrieval if math.isfinite(retrieval) else 0.0
    embedding = embedding if math.isfinite(embedding) else 0.0
    if math.isfinite(archived_cost) and math.isfinite(base_c_llm) and base_c_llm != 0.0:
        return max(0.0, (archived_cost - retrieval - embedding) / base_c_llm)
    return float("nan")


def _candidate_label(row: pd.Series) -> str:
    return f"{row['engine']} / {_short_embedding_label(str(row['embedding_model']))}"


def _decision_latency_targets(*, decision: pd.DataFrame, winners: pd.DataFrame) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(*, scenario: str, role: str, engine: str, embedding: str) -> None:
        key = (scenario, role, engine, embedding)
        if key in seen:
            return
        seen.add(key)
        targets.append({"scenario": scenario, "role": role, "engine": engine, "embedding_model": embedding})

    for _, row in decision.iterrows():
        scenario = str(row["scenario"])
        add(
            scenario=scenario,
            role="strict choice",
            engine=str(row["strict_p99_engine"]),
            embedding=str(row["strict_p99_embedding_model"]),
        )
        add(
            scenario=scenario,
            role="no-p99 cost choice",
            engine=str(row["unconstrained_cost_engine"]),
            embedding=str(row["unconstrained_cost_embedding_model"]),
        )
        add(
            scenario=scenario,
            role="quality-first choice",
            engine=str(row["quality_winner_engine"]),
            embedding=str(row["quality_winner_embedding_model"]),
        )
    for scenario, candidates in _strict_candidate_groups(winners=winners).items():
        if len(candidates) >= 2:
            competitor = candidates.iloc[1]
            add(
                scenario=scenario,
                role="nearest strict-cost competitor",
                engine=str(competitor["engine"]),
                embedding=str(competitor["embedding_model"]),
            )
    return targets


def _latency_boundary(scenario: str) -> str:
    base = "precomputed query vector; timed adapter.query plus top-k materialization; excludes offline embedding"
    if scenario == "s2_streaming_memory":
        return base + "; post-insert probes are measured after insert-plus-flush returns"
    return base


def _source_paths_for_frame(frame: pd.DataFrame) -> str:
    paths: list[str] = []
    for raw in frame.get("__run_path", pd.Series(dtype=object)).dropna().astype(str).tolist():
        run_path = Path(raw)
        for name in ("results.parquet", "config_resolved.yaml", "run_metadata.json"):
            candidate = run_path / name
            if candidate.exists():
                paths.append(_relative_path_string(candidate))
    return "; ".join(sorted(set(paths)))


def _relative_path_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _csv_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    root = str(Path.cwd().resolve())
    for col in safe.columns:
        if safe[col].dtype != object:
            continue
        safe[col] = safe[col].map(lambda value: _strip_workspace_prefix(value, root=root))
    return safe


def _strip_workspace_prefix(value: object, *, root: str) -> object:
    if not isinstance(value, str):
        if isinstance(value, Mapping):
            return {key: _strip_workspace_prefix(item, root=root) for key, item in value.items()}
        if isinstance(value, list):
            return [_strip_workspace_prefix(item, root=root) for item in value]
        return value
    return value.replace(root + "/", "").replace(root, ".")


def _first_config_payload(frame: pd.DataFrame) -> dict[str, Any]:
    for raw in frame.get("__run_path", pd.Series(dtype=object)).dropna().astype(str).tolist():
        payload = _read_yaml_dict(Path(raw) / "config_resolved.yaml")
        if payload:
            return payload
    return {}


def _first_metadata_payload(frame: pd.DataFrame) -> dict[str, Any]:
    for raw in frame.get("__run_path", pd.Series(dtype=object)).dropna().astype(str).tolist():
        payload = _read_json_dict(Path(raw) / "run_metadata.json")
        if payload:
            return payload
    return {}


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _engine_mode(engine: str) -> str:
    return {
        "faiss-cpu": "in-process",
        "lancedb-inproc": "in-process",
        "lancedb-service": "service API",
        "pgvector": "server database",
        "qdrant": "server API",
    }.get(engine, "")


def _engine_process_model(engine: str) -> str:
    return {
        "faiss-cpu": "Python process-local adapter",
        "lancedb-inproc": "Python in-process library with local LanceDB URI",
        "lancedb-service": "external HTTP service contract",
        "pgvector": "PostgreSQL server plus Python adapter",
        "qdrant": "Qdrant server plus Python adapter",
    }.get(engine, "")


def _engine_flush_path(engine: str) -> str:
    return {
        "faiss-cpu": "staged writes become visible after flush_or_commit rebuild/update",
        "lancedb-inproc": "staged writes become visible after flush_or_commit",
        "lancedb-service": "depends on service implementation; no passing local row",
        "pgvector": "writer transaction commit; readers observe committed state",
        "qdrant": "upsert/update/delete requests use wait=true",
    }.get(engine, "")


def _index_search_configuration(config: Mapping[str, Any]) -> str:
    parts: list[str] = []
    index_params = config.get("index_params")
    search_sweep = config.get("search_sweep")
    if isinstance(index_params, Mapping) and index_params:
        parts.append(f"index_params={json.dumps(dict(index_params), sort_keys=True)}")
    if isinstance(search_sweep, list) and search_sweep:
        parts.append(f"search_sweep={json.dumps(search_sweep, sort_keys=True)}")
    top_k = config.get("top_k")
    if top_k is not None:
        parts.append(f"top_k={top_k}")
    return "; ".join(parts) if parts else "default adapter configuration"


def _engine_config_source_path(*, engine_rows: pd.DataFrame, behavior_card: str) -> str:
    paths = []
    if behavior_card:
        paths.append(str(Path("docs/behavior") / behavior_card))
    paths.append("artifacts/conformance/conformance_matrix.csv")
    for raw in engine_rows.get("__run_path", pd.Series(dtype=object)).dropna().astype(str).tolist()[:1]:
        run_path = Path(raw)
        for name in ("config_resolved.yaml", "run_metadata.json"):
            candidate = run_path / name
            if candidate.exists():
                paths.append(_relative_path_string(candidate))
    return "; ".join(sorted(set(paths)))


def _s3_all_evidence_targets(*, winners: pd.DataFrame, decision: pd.DataFrame) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(role: str, engine: str, embedding: str) -> None:
        key = (role, engine, embedding)
        if key in seen:
            return
        seen.add(key)
        targets.append({"role": role, "engine": engine, "embedding_model": embedding})

    s3_decision = decision.loc[decision["scenario"].astype(str) == "s3_multi_hop"]
    if not s3_decision.empty:
        row = s3_decision.iloc[0]
        add("strict choice", str(row["strict_p99_engine"]), str(row["strict_p99_embedding_model"]))
        add("quality-first archived row", str(row["quality_winner_engine"]), str(row["quality_winner_embedding_model"]))
        add("no-p99 cost row", str(row["unconstrained_cost_engine"]), str(row["unconstrained_cost_embedding_model"]))

    s3 = winners.loc[(winners["scenario"].astype(str) == "s3_multi_hop") & (winners["budget_level"].astype(str) == "b2")]
    if not s3.empty:
        candidates = _aggregate_decision_candidates(preferred=s3)
        fast = candidates.loc[pd.to_numeric(candidates["p99_ms_max"], errors="coerce") <= _MVD_P99_MAX_MS_THRESHOLD]
        if not fast.empty:
            high_quality = fast.sort_values(
                ["primary_quality_value", "p99_ms_max", "task_cost_est"],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]
            add("high-quality strict-latency row", str(high_quality["engine"]), str(high_quality["embedding_model"]))
        strict_candidates = _strict_candidate_groups(winners=winners).get("s3_multi_hop", pd.DataFrame())
        if len(strict_candidates) >= 2:
            competitor = strict_candidates.iloc[1]
            add("next strict-cost candidate", str(competitor["engine"]), str(competitor["embedding_model"]))
    return targets


def _load_observations_from_frame(*, selected: pd.DataFrame, include_run_dir_glob: bool) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for raw_path in selected.get("observation_path", pd.Series(dtype=object)).dropna().astype(str).tolist():
        stripped = raw_path.strip()
        if stripped:
            paths.add(Path(stripped))
    if include_run_dir_glob:
        for raw in selected.get("__run_path", pd.Series(dtype=object)).dropna().astype(str).tolist():
            run_path = Path(raw)
            paths.update(run_path.glob("logs/observations/*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            continue
    return rows


def _neurips_main_results_table(*, frame: pd.DataFrame, winners: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    deployment = _minimum_viable_deployment_table(winners=winners)
    rows: list[dict[str, Any]] = []
    for _, choice in deployment.iterrows():
        scenario = str(choice["workload_type"])
        engine = str(choice["minimum_engine"])
        embedding = str(choice["recommended_embedding_tier"])
        scenario_frame = frame.loc[
            (frame["scenario"].astype(str) == scenario)
            & (frame["engine"].astype(str) == engine)
            & (frame["embedding_model"].astype(str) == embedding)
        ].copy()
        b2_frame = scenario_frame.loc[scenario_frame["budget_level"].astype(str) == "b2"].copy()
        selected = b2_frame if not b2_frame.empty else scenario_frame

        metric = str(selected["primary_quality_metric"].dropna().iloc[0]) if not selected.empty else ""
        aggregate_quality_values = pd.to_numeric(selected["primary_quality_value"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        quality_values, quality_ci_method = _quality_observation_values(selected=selected, metric=metric)
        if quality_values.size > 0 and aggregate_quality_values.size > 0:
            aggregate_mean = float(np.mean(aggregate_quality_values))
            observation_mean = float(np.mean(quality_values))
            if not np.isclose(observation_mean, aggregate_mean, atol=1e-9):
                quality_values = aggregate_quality_values
                quality_ci_method = (
                    "aggregate-row bootstrap; archived query observations ignored "
                    "because their mean does not match archived result rows"
                )
        if quality_values.size == 0:
            quality_values = aggregate_quality_values
            quality_ci_method = "aggregate-row bootstrap; query-level observations not present in archived results"
        quality_mean, quality_low, quality_high = _bootstrap_mean_ci(quality_values)
        p99_values = pd.to_numeric(selected.get("p99_ms", pd.Series(dtype=float)), errors="coerce").dropna()
        task_cost_values = pd.to_numeric(selected.get("task_cost_est", pd.Series(dtype=float)), errors="coerce").dropna()
        row: dict[str, Any] = {
            "scenario": scenario,
            "engine": engine,
            "embedding_model": embedding,
            "primary_quality_metric": metric,
            "primary_quality_mean": quality_mean,
            "primary_quality_ci95_low": quality_low,
            "primary_quality_ci95_high": quality_high,
            "primary_quality_ci_method": quality_ci_method,
            "primary_quality_samples": int(len(quality_values)),
            "p99_ms_mean": float(p99_values.mean()) if not p99_values.empty else float("nan"),
            "p99_ms_max": float(p99_values.max()) if not p99_values.empty else float("nan"),
            "task_cost_est_mean": float(task_cost_values.mean()) if not task_cost_values.empty else float("nan"),
            "mvd_p99_max_threshold_ms": _MVD_P99_MAX_MS_THRESHOLD,
        }
        if scenario == "s2_streaming_memory":
            row.update(_s2_post_insert_ci_fields(selected=selected))
        else:
            row.update(
                {
                    "post_insert_hit_at_10_1s_mean": float("nan"),
                    "post_insert_hit_at_10_1s_ci95_low": float("nan"),
                    "post_insert_hit_at_10_1s_ci95_high": float("nan"),
                    "post_insert_hit_at_10_5s_mean": float("nan"),
                    "post_insert_hit_at_10_5s_ci95_low": float("nan"),
                    "post_insert_hit_at_10_5s_ci95_high": float("nan"),
                    "post_insert_event_count": 0,
                    "post_insert_ci_method": "",
                }
            )
        row.update(_stability_fields(stability=stability, scenario=scenario))
        rows.append(row)
    columns = [
        "scenario",
        "engine",
        "embedding_model",
        "primary_quality_metric",
        "primary_quality_mean",
        "primary_quality_ci95_low",
        "primary_quality_ci95_high",
        "primary_quality_ci_method",
        "primary_quality_samples",
        "post_insert_hit_at_10_1s_mean",
        "post_insert_hit_at_10_1s_ci95_low",
        "post_insert_hit_at_10_1s_ci95_high",
        "post_insert_hit_at_10_5s_mean",
        "post_insert_hit_at_10_5s_ci95_low",
        "post_insert_hit_at_10_5s_ci95_high",
        "post_insert_event_count",
        "post_insert_ci_method",
        "p99_ms_mean",
        "p99_ms_max",
        "task_cost_est_mean",
        "mvd_p99_max_threshold_ms",
        "spearman_b0_b2",
        "top1_agreement_b0_b2",
        "top2_agreement_b0_b2",
        "decision_stability_note",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("scenario", kind="stable").reset_index(drop=True)


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if finite.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(_BOOTSTRAP_SEED + int(finite.size))
    sample_idx = rng.integers(0, finite.size, size=(_BOOTSTRAP_RESAMPLES, finite.size))
    boot_means = finite[sample_idx].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _quality_observation_values(*, selected: pd.DataFrame, metric: str) -> tuple[np.ndarray, str]:
    observations = _load_selected_observations(selected=selected)
    if not observations:
        return np.asarray([], dtype=np.float64), ""
    metric_col = _observation_metric_column(metric)
    values = [
        _safe_float(row.get(metric_col))
        for row in observations
        if str(row.get("observation_type") or "") == "quality"
    ]
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return finite, ""
    return finite, f"query-level bootstrap from {int(finite.size)} archived measured-query observations"


def _load_selected_observations(*, selected: pd.DataFrame) -> list[dict[str, Any]]:
    if selected.empty or "observation_path" not in selected.columns:
        return []
    rows: list[dict[str, Any]] = []
    paths = [
        str(path).strip()
        for path in selected["observation_path"].dropna().astype(str).tolist()
        if str(path).strip()
    ]
    for raw_path in sorted(set(paths)):
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            continue
    return rows


def _observation_metric_column(metric: str) -> str:
    return {
        "ndcg_at_10": "ndcg_at_10",
        "evidence_coverage@10": "evidence_coverage_at_10",
    }.get(metric, metric)


def _s2_post_insert_ci_fields(*, selected: pd.DataFrame) -> dict[str, Any]:
    observations = [
        row
        for row in _load_selected_observations(selected=selected)
        if str(row.get("observation_type") or "") == "freshness"
    ]
    if observations:
        fields: dict[str, Any] = {
            "post_insert_event_count": len(observations),
            "post_insert_ci_method": "Wilson binomial CI from archived per-event post-insert observations",
        }
        for col in ("freshness_hit_at_1s", "freshness_hit_at_5s"):
            hits = [
                _safe_float(row.get(col))
                for row in observations
                if math.isfinite(_safe_float(row.get(col)))
            ]
            rate = float(np.mean(np.asarray(hits, dtype=np.float64))) if hits else float("nan")
            low, high = _wilson_ci(rate=rate, n=len(hits))
            output_col = _REPORT_COLUMN_ALIASES[col]
            fields[f"{output_col}_mean"] = rate
            fields[f"{output_col}_ci95_low"] = low
            fields[f"{output_col}_ci95_high"] = high
        return fields

    event_counts = pd.to_numeric(selected.get("event_count", pd.Series(dtype=float)), errors="coerce").dropna()
    event_count = int(event_counts.max()) if not event_counts.empty else 0
    fields: dict[str, Any] = {
        "post_insert_event_count": event_count,
        "post_insert_ci_method": "Wilson binomial CI from archived post-insert hit rate and event_count; repeated runs are not counted as independent events",
    }
    for col in ("freshness_hit_at_1s", "freshness_hit_at_5s"):
        rates = pd.to_numeric(selected.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        rate = float(rates.mean()) if not rates.empty else float("nan")
        low, high = _wilson_ci(rate=rate, n=event_count)
        output_col = _REPORT_COLUMN_ALIASES[col]
        fields[f"{output_col}_mean"] = rate
        fields[f"{output_col}_ci95_low"] = low
        fields[f"{output_col}_ci95_high"] = high
    return fields


def _wilson_ci(*, rate: float, n: int) -> tuple[float, float]:
    if n <= 0 or not math.isfinite(rate):
        return float("nan"), float("nan")
    p = min(max(rate, 0.0), 1.0)
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _stability_fields(*, stability: pd.DataFrame, scenario: str) -> dict[str, Any]:
    pair = stability.loc[
        (stability["scenario"].astype(str) == scenario)
        & (stability["budget_pair"].astype(str) == "b0->b2")
    ]
    if pair.empty:
        return {
            "spearman_b0_b2": float("nan"),
            "top1_agreement_b0_b2": float("nan"),
            "top2_agreement_b0_b2": float("nan"),
            "decision_stability_note": "b0->b2 overlap unavailable",
        }
    row = pair.iloc[0]
    spearman = float(row["spearman_rho"])
    top1 = float(row["top1_agreement"])
    top2 = float(row["top2_agreement"])
    if top1 < 1.0:
        note = "top-1 changed between b0 and b2"
    elif math.isfinite(spearman) and spearman < 0.8:
        note = "top-1 stable despite full-rank noise"
    else:
        note = "top-1 and full-rank ordering broadly aligned"
    return {
        "spearman_b0_b2": spearman,
        "top1_agreement_b0_b2": top1,
        "top2_agreement_b0_b2": top2,
        "decision_stability_note": note,
    }


def _neurips_main_results_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Portable benchmark main results. Quality and S2 post-insert top-10 retrievability report 95\\% confidence intervals; p99 reports the maximum observed tail latency across selected runs.}",
        "\\label{tab:portable-main-results}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Workload & Engine & Emb. & Quality & post\\_insert\\_hit@10,5s & p99 max & Stability \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        workload = _latex_escape(_short_scenario_label(str(row["scenario"])))
        engine = _latex_escape(str(row["engine"]))
        embedding = _latex_escape(_short_embedding_label(str(row["embedding_model"])))
        metric = _latex_escape(_short_metric_label(str(row["primary_quality_metric"])))
        quality = _format_ci(
            mean=_safe_float(row["primary_quality_mean"]),
            low=_safe_float(row["primary_quality_ci95_low"]),
            high=_safe_float(row["primary_quality_ci95_high"]),
        )
        quality_cell = f"{metric} {quality}"
        fresh = _format_ci(
            mean=_safe_float(row["post_insert_hit_at_10_5s_mean"]),
            low=_safe_float(row["post_insert_hit_at_10_5s_ci95_low"]),
            high=_safe_float(row["post_insert_hit_at_10_5s_ci95_high"]),
            empty="--",
        )
        p99 = _safe_float(row["p99_ms_max"])
        p99_cell = "--" if not math.isfinite(p99) else f"{p99:.1f} ms"
        stability = _latex_escape(_short_stability_note(str(row["decision_stability_note"])))
        lines.append(
            f"{workload} & {engine} & {embedding} & {quality_cell} & {fresh} & {p99_cell} & {stability} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _portable_decision_table_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Deployment decisions under strict latency, unconstrained cost, and quality-first objectives. Strict latency uses a 200 ms worst-case p99 threshold.}",
        "\\label{tab:portable-decision-table}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllcc}",
        "\\toprule",
        "Workload & Strict p99 & No-p99 cost & Quality-first & $\\rho$ B0--B2 & Stability \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        workload = _latex_escape(_short_scenario_label(str(row["scenario"])))
        strict = _decision_choice_label(
            engine=str(row["strict_p99_engine"]),
            embedding=str(row["strict_p99_embedding_model"]),
        )
        unconstrained = _decision_choice_label(
            engine=str(row["unconstrained_cost_engine"]),
            embedding=str(row["unconstrained_cost_embedding_model"]),
        )
        quality = _decision_choice_label(
            engine=str(row["quality_winner_engine"]),
            embedding=str(row["quality_winner_embedding_model"]),
        )
        rho = _safe_float(row["spearman_b0_b2"])
        rho_cell = "--" if not math.isfinite(rho) else f"{rho:.2f}"
        stability = _latex_escape(_short_stability_note(str(row["decision_stability_note"])))
        lines.append(
            f"{workload} & {strict} & {unconstrained} & {quality} & {rho_cell} & {stability} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_error_ablation_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Protocol components whose omission changes or weakens deployment conclusions.}",
        "\\label{tab:decision-error-ablation}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Missing component & Wrong conclusion caused by omission & Evidence \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_latex_escape(str(row['missing_protocol_component']))} & "
            f"{_latex_escape(str(row['wrong_conclusion_caused_by_omission']))} & "
            f"{_latex_escape(str(row['manuscript_evidence']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _cost_formula_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Task-cost estimate terms and deterministic value sources.}",
        "\\label{tab:cost-formula}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Term & Meaning & Unit & Value source \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_latex_escape(str(row['term']))} & {_latex_escape(str(row['meaning']))} & "
            f"{_latex_escape(str(row['unit']))} & {_latex_escape(str(row['value_source']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _cost_sensitivity_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Strict-choice cost sensitivity to input-token cost multipliers.}",
        "\\label{tab:cost-sensitivity}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrrrl}",
        "\\toprule",
        "Workload & Role & Candidate & Mult. & Tokens & Cost & Changes? & Selected \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        candidate = f"{row['engine']} / {_short_embedding_label(str(row['embedding_model']))}"
        lines.append(
            f"{_latex_escape(_short_scenario_label(str(row['workload'])))} & "
            f"{_latex_escape(str(row['candidate_role']))} & {_latex_escape(candidate)} & "
            f"{_safe_float(row['c_llm_in_multiplier']):.1f} & "
            f"{_safe_float(row['avg_retrieved_input_tokens']):.0f} & "
            f"{_safe_float(row['sensitivity_task_cost_est']):.3f} & "
            f"{'yes' if bool(row['selection_changes_from_main']) else 'no'} & "
            f"{_latex_escape(str(row['selected_config_at_multiplier']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _latency_distribution_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Latency distributions for strict choices and objective-sensitivity rows.}",
        "\\label{tab:latency-distribution}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Workload & Role & Candidate & p50 & p95 & p99 & p99 max & Obs. \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        candidate = f"{row['engine']} / {_short_embedding_label(str(row['embedding_model']))}"
        lines.append(
            f"{_latex_escape(_short_scenario_label(str(row['workload'])))} & {_latex_escape(str(row['row_role']))} & "
            f"{_latex_escape(candidate)} & {_safe_float(row['p50_ms']):.1f} & {_safe_float(row['p95_ms']):.1f} & "
            f"{_safe_float(row['p99_ms']):.1f} & {_safe_float(row['p99_max_ms']):.1f} & "
            f"{int(_safe_float(row['latency_observations']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _engine_configuration_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Engine modes and configuration sources for the reported matrix.}",
        "\\label{tab:engine-configuration}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Engine & Mode & Version & Dim. & Process model & Reported? \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{_latex_escape(str(row['engine']))} & {_latex_escape(str(row['mode']))} & "
            f"{_latex_escape(str(row['version']))} & {_latex_escape(str(row['embedding_dimension']))} & "
            f"{_latex_escape(str(row['process_model']))} & "
            f"{'yes' if bool(row['included_in_reported_matrix']) else 'no'} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _support_table_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Conformance and behavior-card support for paper-facing reportability.}",
        "\\label{tab:portable-support}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Engine & Behavior card & Conformance & Reported? & Exclusion reason \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        reason = str(row.get("exclusion_reason", ""))
        lines.append(
            f"{_latex_escape(str(row['engine']))} & {_latex_escape(str(row['behavior_card']))} & "
            f"{_latex_escape(str(row['conformance_statuses']))} & "
            f"{'yes' if bool(row['included_in_report']) else 'no'} & "
            f"{_latex_escape(reason)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _s3_all_evidence_hit_latex(*, table: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by maxionbench.reports.portable_exports.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{S3 all-evidence hit@10 audit rows. Query-level rows come from archived observations when present and targeted matched-audit observations otherwise.}",
        "\\label{tab:s3-all-evidence-hit}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Role & Engine & Embedding & Cov@10 & All-hit@10 & Query rows \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        all_hit = _safe_float(row["all_evidence_hit_at_10"])
        all_hit_cell = "--" if not math.isfinite(all_hit) else f"{all_hit:.3f}"
        coverage = _safe_float(row["evidence_coverage_at_10"])
        coverage_cell = "--" if not math.isfinite(coverage) else f"{coverage:.3f}"
        lines.append(
            f"{_latex_escape(str(row['row_role']))} & {_latex_escape(str(row['engine']))} & "
            f"{_latex_escape(_short_embedding_label(str(row['embedding_model'])))} & {coverage_cell} & "
            f"{all_hit_cell} & {int(_safe_float(row['query_level_observations']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _decision_choice_label(*, engine: str, embedding: str) -> str:
    return _latex_escape(f"{engine} / {_short_embedding_label(embedding)}")


def _format_ci(*, mean: float, low: float, high: float, empty: str = "") -> str:
    if not math.isfinite(mean):
        return empty
    if math.isfinite(low) and math.isfinite(high):
        return f"{mean:.3f} ({low:.3f}--{high:.3f})"
    return f"{mean:.3f}"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _short_scenario_label(scenario: str) -> str:
    return {
        "s1_single_hop": "S1 single-hop",
        "s2_streaming_memory": "S2 streaming",
        "s3_multi_hop": "S3 multi-evidence",
    }.get(scenario, scenario)


def _short_embedding_label(embedding: str) -> str:
    if "bge-small" in embedding:
        return "BAAI/bge-small-en-v1.5"
    if "bge-base" in embedding:
        return "BAAI/bge-base-en-v1.5"
    return embedding


def _short_metric_label(metric: str) -> str:
    return {
        "ndcg_at_10": "nDCG@10",
        "evidence_coverage@10": "Cov@10",
    }.get(metric, metric)


def _short_stability_note(note: str) -> str:
    if note == "top-1 stable despite full-rank noise":
        return "top-1 stable"
    if note == "top-1 changed between b0 and b2":
        return "top-1 changed"
    return "aligned"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _support_table(
    *,
    frame: pd.DataFrame,
    winners: pd.DataFrame,
    conformance_matrix_path: Path | None,
    behavior_dir: Path | None,
) -> pd.DataFrame:
    conformance_status_by_adapter = _load_conformance_statuses(conformance_matrix_path)
    behavior_root = behavior_dir.resolve() if behavior_dir is not None else None
    reported_engines = {str(value).strip() for value in winners.get("engine", pd.Series(dtype=object)).tolist() if str(value).strip()}
    observed_engines = {str(value).strip() for value in frame.get("engine", pd.Series(dtype=object)).tolist() if str(value).strip()}

    rows: list[dict[str, Any]] = []
    for adapter in REQUIRED_ADAPTERS:
        behavior_card = str(BEHAVIOR_CARD_BY_ADAPTER.get(adapter, ""))
        behavior_card_present = bool(behavior_root and behavior_card and (behavior_root / behavior_card).exists())
        statuses = conformance_status_by_adapter.get(adapter, [])
        has_pass = "pass" in statuses
        reportable = bool(has_pass and behavior_card_present)
        included_in_report = bool(reportable and adapter in reported_engines)
        observed_in_runs = adapter in observed_engines

        exclusion_reason = ""
        if included_in_report:
            exclusion_reason = ""
        elif not statuses:
            exclusion_reason = "missing conformance row"
        elif not has_pass:
            exclusion_reason = f"conformance status {','.join(statuses)}"
        elif not behavior_card_present:
            exclusion_reason = f"missing behavior card {behavior_card}"
        elif observed_in_runs and adapter not in reported_engines:
            exclusion_reason = "observed in runs but filtered from reportable winners"
        else:
            exclusion_reason = "not present in reported results"

        rows.append(
            {
                "engine": adapter,
                "behavior_card": behavior_card,
                "behavior_card_present": behavior_card_present,
                "conformance_statuses": ",".join(statuses),
                "reportable": reportable,
                "included_in_report": included_in_report,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows).sort_values("engine", kind="stable").reset_index(drop=True)


def _load_conformance_statuses(conformance_matrix_path: Path | None) -> dict[str, list[str]]:
    if conformance_matrix_path is None:
        return {}
    path = conformance_matrix_path.resolve()
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if not {"adapter", "status"}.issubset(frame.columns):
        return {}
    normalized = frame.copy()
    normalized["adapter"] = normalized["adapter"].fillna("").astype(str).str.strip()
    normalized["status"] = normalized["status"].fillna("").astype(str).str.strip().str.lower()
    rows: dict[str, list[str]] = {}
    for adapter, group in normalized.groupby("adapter", dropna=False):
        key = str(adapter).strip()
        if not key:
            continue
        statuses = sorted({str(value).strip() for value in group["status"].tolist() if str(value).strip()})
        rows[key] = statuses
    return rows


def _resolve_reportability_inputs(
    *,
    conformance_matrix_path: Path | None,
    behavior_dir: Path | None,
) -> tuple[Path, Path]:
    if conformance_matrix_path is None:
        raise ValueError("portable paper-facing reports require --conformance-matrix")
    if behavior_dir is None:
        raise ValueError("portable paper-facing reports require --behavior-dir")
    resolved_conformance_matrix_path = conformance_matrix_path.resolve()
    if not resolved_conformance_matrix_path.exists():
        raise FileNotFoundError(f"conformance matrix not found: {resolved_conformance_matrix_path}")
    resolved_behavior_dir = behavior_dir.resolve()
    if not resolved_behavior_dir.exists():
        raise FileNotFoundError(f"behavior directory not found: {resolved_behavior_dir}")
    return resolved_conformance_matrix_path, resolved_behavior_dir


def _reportability_by_adapter(
    *,
    conformance_matrix_path: Path,
    behavior_dir: Path,
) -> dict[str, dict[str, bool | str]]:
    statuses_by_adapter = _load_conformance_statuses(conformance_matrix_path)
    reportability: dict[str, dict[str, bool | str]] = {}
    for adapter in sorted(set(REQUIRED_ADAPTERS) | set(statuses_by_adapter)):
        behavior_card = str(BEHAVIOR_CARD_BY_ADAPTER.get(adapter, ""))
        behavior_card_present = True if not behavior_card else bool((behavior_dir / behavior_card).exists())
        statuses = statuses_by_adapter.get(adapter, [])
        reportability[adapter] = {
            "reportable": bool("pass" in statuses and behavior_card_present),
            "behavior_card": behavior_card,
        }
    return reportability


def _paper_figure(*, height_in: float | None = None) -> tuple[Any, Any]:
    return plt.subplots(
        figsize=(PANEL_WIDTH_IN, height_in or PANEL_HEIGHT_IN),
        dpi=DPI,
        constrained_layout=True,
    )


def _save_paper_figure(*, fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=DPI, format="png", facecolor=FIGURE_FACE_COLOR, edgecolor="none")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", facecolor=FIGURE_FACE_COLOR, edgecolor="none")


def _plot_task_cost_by_budget(*, ax: Any, winners: pd.DataFrame) -> None:
    if winners.empty:
        _draw_placeholder(ax=ax, message="No portable winners available")
        return
    summary = winners.sort_values(["scenario", "budget_sort", "rank_within_budget"], kind="stable").groupby(
        ["scenario", "budget_level"], as_index=False
    ).first()
    labels = [_scenario_budget_tick_label(str(row["scenario"]), str(row["budget_level"])) for _, row in summary.iterrows()]
    colors = [_engine_color(str(row["engine"])) for _, row in summary.iterrows()]
    bars = ax.bar(np.arange(len(summary)), summary["task_cost_est"].astype(float), color=colors, width=0.72)
    ax.bar_label(
        bars,
        labels=[f"{value:.0f}" for value in summary["task_cost_est"].astype(float)],
        padding=2,
        fontsize=max(10, FONT_SIZE - 2),
    )
    ax.set_xticks(np.arange(len(summary)), labels=labels, rotation=0, ha="center")
    ax.set_ylabel("Task cost estimate")
    ax.set_xlabel("Selected workload-budget winner")
    ax.margins(y=0.14)
    ax.grid(axis="y", alpha=0.35)
    _style_axis(ax)


def _plot_budget_stability(*, ax: Any, stability: pd.DataFrame) -> None:
    if stability.empty:
        _draw_placeholder(ax=ax, message="No cross-budget overlap available")
        return
    ordered = stability.loc[stability["budget_pair"].astype(str) == "b0->b2"].copy()
    if ordered.empty:
        ordered = stability.copy()
    ordered["scenario_label"] = ordered["scenario"].astype(str).map(_short_scenario_label)
    ordered = ordered.sort_values(["scenario"], kind="stable").reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.24
    bars = [
        ax.bar(x - width, ordered["spearman_rho"].astype(float), width=width, label="Rank corr.", color=ENGINE_PALETTE[0]),
        ax.bar(x, ordered["top1_agreement"].astype(float), width=width, label="Top-1", color=ENGINE_PALETTE[1]),
        ax.bar(x + width, ordered["top2_agreement"].astype(float), width=width, label="Top-2", color=ENGINE_PALETTE[2]),
    ]
    for group in bars:
        ax.bar_label(
            group,
            labels=[f"{bar.get_height():.2f}" for bar in group],
            padding=2,
            fontsize=max(10, FONT_SIZE - 2),
        )
    labels = ordered["scenario_label"].astype(str).tolist()
    ax.set_xticks(x, labels=labels, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.16)
    ax.set_ylabel("B0 -> B2 stability")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, handlelength=1.2)
    _style_axis(ax)


def _plot_s2_post_insert_retrievability(*, ax: Any, winners: pd.DataFrame) -> None:
    s2 = winners.loc[winners["scenario"] == "s2_streaming_memory"].copy()
    if s2.empty:
        _draw_placeholder(ax=ax, message="No S2 post-insert rows available")
        return
    s2 = s2.sort_values(["budget_sort", "rank_within_budget", "engine"], kind="stable").groupby("engine", as_index=False).first()
    x = np.arange(len(s2))
    width = 0.30
    bars_1s = ax.bar(
        x - width / 2,
        s2["freshness_hit_at_1s"].astype(float),
        width=width,
        label="post-insert@1s",
        color=ENGINE_PALETTE[0],
    )
    bars_5s = ax.bar(
        x + width / 2,
        s2["freshness_hit_at_5s"].astype(float),
        width=width,
        label="post-insert@5s",
        color=ENGINE_PALETTE[1],
    )
    for group in (bars_1s, bars_5s):
        ax.bar_label(
            group,
            labels=[f"{bar.get_height():.2f}" for bar in group],
            padding=2,
            fontsize=max(10, FONT_SIZE - 2),
        )
    ax.set_xticks(x, labels=s2["engine"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Top-10 retrievability")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2)
    _style_axis(ax)


def _plot_mvd_sensitivity(*, ax: Any, sensitivity: pd.DataFrame) -> None:
    if sensitivity.empty:
        _draw_placeholder(ax=ax, message="No MVD sensitivity rows available")
        return
    thresholds = ["100.0", "200.0", "500.0", "none"]
    threshold_labels = ["100", "200\nmain", "500", "No cap"]
    scenarios = ["s1_single_hop", "s2_streaming_memory", "s3_multi_hop"]
    scenario_labels = [_short_scenario_label(scenario) for scenario in scenarios]
    scenario_to_y = {scenario: idx for idx, scenario in enumerate(scenarios)}
    threshold_to_x = {threshold: idx for idx, threshold in enumerate(thresholds)}
    observed_engines: list[str] = []

    for _, row in sensitivity.iterrows():
        scenario = str(row["workload_type"])
        threshold = str(row["p99_max_threshold_ms"])
        if scenario not in scenario_to_y or threshold not in threshold_to_x:
            continue
        engine = str(row["minimum_engine"])
        if engine not in observed_engines:
            observed_engines.append(engine)
        x = threshold_to_x[threshold]
        y = scenario_to_y[scenario]
        ax.scatter(
            [x],
            [y],
            s=470,
            marker="s",
            color=_engine_color(engine),
            edgecolor=FIGURE_FACE_COLOR,
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            x,
            y,
            _short_engine_label(engine),
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=max(10, FONT_SIZE - 4),
            fontweight="bold",
            zorder=4,
        )

    ax.axvline(1, color=TEXT_COLOR, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xticks(np.arange(len(thresholds)), labels=threshold_labels)
    ax.set_yticks(np.arange(len(scenarios)), labels=scenario_labels)
    ax.set_xlim(-0.75, len(thresholds) - 0.25)
    ax.set_ylim(len(scenarios) - 0.4, -0.6)
    ax.set_xlabel("p99 max latency cap (ms)")
    ax.set_ylabel("Workload")
    ax.grid(axis="x", alpha=0.25)
    for engine in observed_engines:
        ax.scatter([], [], s=110, marker="s", color=_engine_color(engine), label=_short_engine_name(engine))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=max(10, FONT_SIZE - 3),
    )
    _style_axis(ax)


def _quality_floor(scenario: str) -> float:
    if scenario == "s3_multi_hop":
        return 0.30
    return 0.25


def _scenario_budget_tick_label(scenario: str, budget: str) -> str:
    return f"{_short_scenario_code(scenario)}\n{budget}"


def _short_scenario_code(scenario: str) -> str:
    return {
        "s1_single_hop": "S1",
        "s2_streaming_memory": "S2",
        "s3_multi_hop": "S3",
    }.get(scenario, scenario)


def _style_axis(ax: Any) -> None:
    ax.tick_params(axis="both", colors=TEXT_COLOR, length=3, width=0.8)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)


def _engine_color(engine: str) -> str:
    normalized = str(engine).strip().lower()
    fixed = {
        "faiss-cpu": ENGINE_PALETTE[0],
        "lancedb-inproc": ENGINE_PALETTE[1],
        "pgvector": ENGINE_PALETTE[2],
        "qdrant": ENGINE_PALETTE[3],
        "lancedb-service": ENGINE_PALETTE[4],
    }
    if normalized in fixed:
        return fixed[normalized]
    index = sum(ord(char) for char in normalized) % len(ENGINE_PALETTE)
    return ENGINE_PALETTE[index]


def _short_engine_label(engine: str) -> str:
    return {
        "faiss-cpu": "F",
        "lancedb-inproc": "L",
        "pgvector": "PG",
        "qdrant": "Q",
        "lancedb-service": "LS",
    }.get(engine, engine[:2].upper())


def _short_engine_name(engine: str) -> str:
    return {
        "faiss-cpu": "FAISS CPU",
        "lancedb-inproc": "LanceDB inproc",
        "pgvector": "pgvector",
        "qdrant": "Qdrant",
        "lancedb-service": "LanceDB service",
    }.get(engine, engine)


def _extract_search_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _payload_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _spearman_rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    left_centered = left_arr - float(np.mean(left_arr))
    right_centered = right_arr - float(np.mean(right_arr))
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denom)


def _coalesced_float_column(frame: pd.DataFrame, key: str) -> pd.Series:
    direct = pd.to_numeric(frame.get(key, pd.Series(dtype=float)), errors="coerce")
    fallback = frame["__search_payload"].map(lambda payload, item=key: _payload_float(payload, item))
    return direct.where(~direct.isna(), fallback)


def _coalesced_string_column(frame: pd.DataFrame, key: str) -> pd.Series:
    return _normalized_string_series(frame.get(key, pd.Series(dtype=object)))


def _normalized_string_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str)
    return normalized.map(lambda value: "" if value.strip().lower() in {"", "none", "nan"} else value)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.facecolor": FIGURE_FACE_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": FIGURE_FACE_COLOR,
            "savefig.facecolor": FIGURE_FACE_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": plt.cycler(color=ENGINE_PALETTE),
        }
    )


def _write_meta(path: Path, payload: Mapping[str, Any]) -> None:
    path.with_suffix(".meta.json").write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _draw_placeholder(*, ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=TEXT_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
