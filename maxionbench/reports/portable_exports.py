"""Portable-agentic report export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

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
    summary_path = out_dir / "portable_summary.csv"
    summary.to_csv(summary_path, index=False)
    tables.append(summary_path)

    winners = _winner_rows(frame=frame)
    winners_path = out_dir / "portable_winners.csv"
    winners.to_csv(winners_path, index=False)
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

    support = _support_table(
        frame=observed_frame,
        winners=winners,
        conformance_matrix_path=conformance_matrix_path,
        behavior_dir=behavior_dir,
    )
    support_path = out_dir / "portable_support_table.csv"
    support.to_csv(support_path, index=False)
    tables.append(support_path)

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

    freshness_path = out_dir / "portable_s2_freshness.png"
    fig, ax = _paper_figure()
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_s2_freshness(ax=ax, winners=winners)
    _save_paper_figure(fig=fig, path=freshness_path)
    plt.close(fig)
    _write_meta(
        freshness_path,
        {
            "figure_name": "portable_s2_freshness",
            "mode": "portable-agentic",
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "rows_used": int(len(winners.loc[winners["scenario"] == "s2_streaming_memory"])),
        },
    )
    figures.append(freshness_path)

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
            reason_parts.append(f"freshness_hit@5s={float(freshness_val):.3f}")
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
            row.update(_s2_freshness_ci_fields(selected=selected))
        else:
            row.update(
                {
                    "freshness_hit_at_1s_mean": float("nan"),
                    "freshness_hit_at_1s_ci95_low": float("nan"),
                    "freshness_hit_at_1s_ci95_high": float("nan"),
                    "freshness_hit_at_5s_mean": float("nan"),
                    "freshness_hit_at_5s_ci95_low": float("nan"),
                    "freshness_hit_at_5s_ci95_high": float("nan"),
                    "freshness_event_count": 0,
                    "freshness_ci_method": "",
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
        "freshness_hit_at_1s_mean",
        "freshness_hit_at_1s_ci95_low",
        "freshness_hit_at_1s_ci95_high",
        "freshness_hit_at_5s_mean",
        "freshness_hit_at_5s_ci95_low",
        "freshness_hit_at_5s_ci95_high",
        "freshness_event_count",
        "freshness_ci_method",
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


def _s2_freshness_ci_fields(*, selected: pd.DataFrame) -> dict[str, Any]:
    observations = [
        row
        for row in _load_selected_observations(selected=selected)
        if str(row.get("observation_type") or "") == "freshness"
    ]
    if observations:
        fields: dict[str, Any] = {
            "freshness_event_count": len(observations),
            "freshness_ci_method": "Wilson binomial CI from archived per-event freshness observations",
        }
        for col in ("freshness_hit_at_1s", "freshness_hit_at_5s"):
            hits = [
                _safe_float(row.get(col))
                for row in observations
                if math.isfinite(_safe_float(row.get(col)))
            ]
            rate = float(np.mean(np.asarray(hits, dtype=np.float64))) if hits else float("nan")
            low, high = _wilson_ci(rate=rate, n=len(hits))
            fields[f"{col}_mean"] = rate
            fields[f"{col}_ci95_low"] = low
            fields[f"{col}_ci95_high"] = high
        return fields

    event_counts = pd.to_numeric(selected.get("event_count", pd.Series(dtype=float)), errors="coerce").dropna()
    event_count = int(event_counts.max()) if not event_counts.empty else 0
    fields: dict[str, Any] = {
        "freshness_event_count": event_count,
        "freshness_ci_method": "Wilson binomial CI from archived hit rate and event_count; repeated runs are not counted as independent events",
    }
    for col in ("freshness_hit_at_1s", "freshness_hit_at_5s"):
        rates = pd.to_numeric(selected.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        rate = float(rates.mean()) if not rates.empty else float("nan")
        low, high = _wilson_ci(rate=rate, n=event_count)
        fields[f"{col}_mean"] = rate
        fields[f"{col}_ci95_low"] = low
        fields[f"{col}_ci95_high"] = high
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
        "\\caption{Portable benchmark main results. Quality and freshness report 95\\% confidence intervals; p99 reports the maximum observed tail latency across selected runs.}",
        "\\label{tab:portable-main-results}",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Workload & Engine & Emb. & Quality & Fresh@5 & p99 max & Stability \\\\",
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
            mean=_safe_float(row["freshness_hit_at_5s_mean"]),
            low=_safe_float(row["freshness_hit_at_5s_ci95_low"]),
            high=_safe_float(row["freshness_hit_at_5s_ci95_high"]),
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
            "\\end{table}",
            "",
        ]
    )
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
        "s3_multi_hop": "S3 multi-hop",
    }.get(scenario, scenario)


def _short_embedding_label(embedding: str) -> str:
    if "bge-small" in embedding:
        return "bge-small"
    if "bge-base" in embedding:
        return "bge-base"
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
    ax.bar_label(bars, labels=[f"{value:.0f}" for value in summary["task_cost_est"].astype(float)], padding=2, fontsize=8)
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
        ax.bar_label(group, labels=[f"{bar.get_height():.2f}" for bar in group], padding=2, fontsize=8)
    labels = ordered["scenario_label"].astype(str).tolist()
    ax.set_xticks(x, labels=labels, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.16)
    ax.set_ylabel("B0 -> B2 stability")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, handlelength=1.2)
    _style_axis(ax)


def _plot_s2_freshness(*, ax: Any, winners: pd.DataFrame) -> None:
    s2 = winners.loc[winners["scenario"] == "s2_streaming_memory"].copy()
    if s2.empty:
        _draw_placeholder(ax=ax, message="No S2 freshness rows available")
        return
    s2 = s2.sort_values(["budget_sort", "rank_within_budget", "engine"], kind="stable").groupby("engine", as_index=False).first()
    x = np.arange(len(s2))
    width = 0.30
    bars_1s = ax.bar(x - width / 2, s2["freshness_hit_at_1s"].astype(float), width=width, label="hit@1s", color=ENGINE_PALETTE[0])
    bars_5s = ax.bar(x + width / 2, s2["freshness_hit_at_5s"].astype(float), width=width, label="hit@5s", color=ENGINE_PALETTE[1])
    for group in (bars_1s, bars_5s):
        ax.bar_label(group, labels=[f"{bar.get_height():.2f}" for bar in group], padding=2, fontsize=8)
    ax.set_xticks(x, labels=s2["engine"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Freshness hit rate")
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
            fontsize=8,
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
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=9)
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
