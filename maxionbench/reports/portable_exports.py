"""Portable-agentic report export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .plots import DPI, FIGURE_FACE_COLOR, FONT_SIZE, PANEL_PX, STYLE_VERSION, TEXT_COLOR, GRID_COLOR, ENGINE_PALETTE, load_results

import matplotlib.pyplot as plt


_PORTABLE_SCENARIOS = {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}
_BUDGET_ORDER = {"b0": 0, "b1": 1, "b2": 2}
_BUDGET_PAIRS = [("b0", "b1"), ("b1", "b2"), ("b0", "b2")]


def generate_portable_report_bundle(*, input_dir: Path, out_dir: Path) -> dict[str, list[Path]]:
    frame = load_results(input_dir)
    portable = _extract_portable_frame(frame=frame)
    if portable.empty:
        raise RuntimeError(
            f"no portable-agentic results found under {input_dir}; expected scenarios {sorted(_PORTABLE_SCENARIOS)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = _export_portable_tables(frame=portable, out_dir=out_dir)
    figures = _export_portable_figures(frame=portable, out_dir=out_dir)
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


def _export_portable_tables(*, frame: pd.DataFrame, out_dir: Path) -> list[Path]:
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
        ]
    ].copy()
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

    meta_path = out_dir / "portable_summary.meta.json"
    meta_payload = {
        "mode": "portable-agentic",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "rows_total": int(len(frame)),
        "winner_rows": int(len(winners)),
        "table_names": [path.name for path in tables],
        "budgets": sorted({str(value) for value in frame["budget_level"].tolist() if str(value)}),
        "scenarios": sorted({str(value) for value in frame["scenario"].tolist() if str(value)}),
        "engines": sorted({str(value) for value in frame["engine"].tolist() if str(value)}),
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
    fig, ax = plt.subplots(figsize=(PANEL_PX / DPI, PANEL_PX / DPI), dpi=DPI, constrained_layout=True)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_task_cost_by_budget(ax=ax, winners=winners)
    fig.savefig(task_cost_path, dpi=DPI, format="png", facecolor=FIGURE_FACE_COLOR, edgecolor="none")
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
    fig, ax = plt.subplots(figsize=(PANEL_PX / DPI, PANEL_PX / DPI), dpi=DPI, constrained_layout=True)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_budget_stability(ax=ax, stability=stability)
    fig.savefig(stability_path, dpi=DPI, format="png", facecolor=FIGURE_FACE_COLOR, edgecolor="none")
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
    fig, ax = plt.subplots(figsize=(PANEL_PX / DPI, PANEL_PX / DPI), dpi=DPI, constrained_layout=True)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    _plot_s2_freshness(ax=ax, winners=winners)
    fig.savefig(freshness_path, dpi=DPI, format="png", facecolor=FIGURE_FACE_COLOR, edgecolor="none")
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


def _minimum_viable_deployment_table(*, winners: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in winners.groupby("scenario", dropna=False):
        preferred = scenario_frame.loc[scenario_frame["budget_level"] == "b2"]
        if preferred.empty:
            preferred = scenario_frame.sort_values(["budget_sort", "rank_within_budget"], kind="stable").tail(1)
        best = preferred.sort_values(["task_cost_est", "p99_ms", "qps"], ascending=[True, True, False], kind="stable").iloc[0]
        reason = f"{best['primary_quality_metric']}={float(best['primary_quality_value']):.3f}, p99={float(best['p99_ms']):.1f}ms, task_cost={float(best['task_cost_est']):.6f}"
        rows.append(
            {
                "workload_type": str(scenario),
                "minimum_engine": str(best["engine"]),
                "recommended_embedding_tier": str(best["embedding_model"]),
                "why": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("workload_type", kind="stable").reset_index(drop=True)


def _plot_task_cost_by_budget(*, ax: Any, winners: pd.DataFrame) -> None:
    if winners.empty:
        _draw_placeholder(ax=ax, message="No portable winners available")
        return
    summary = winners.sort_values(["scenario", "budget_sort", "rank_within_budget"], kind="stable").groupby(
        ["scenario", "budget_level"], as_index=False
    ).first()
    labels = [f"{row['scenario']}\n{row['budget_level']}" for _, row in summary.iterrows()]
    colors = [_engine_color(str(row["engine"])) for _, row in summary.iterrows()]
    ax.bar(np.arange(len(summary)), summary["task_cost_est"].astype(float), color=colors)
    ax.set_xticks(np.arange(len(summary)), labels=labels, rotation=30, ha="right")
    ax.set_ylabel("Task Cost Est")
    ax.set_title("Portable Winners by Budget")
    ax.grid(axis="y", alpha=0.4)


def _plot_budget_stability(*, ax: Any, stability: pd.DataFrame) -> None:
    if stability.empty:
        _draw_placeholder(ax=ax, message="No cross-budget overlap available")
        return
    x = np.arange(len(stability))
    ax.plot(x, stability["spearman_rho"].astype(float), marker="o", label="Spearman rho", color=ENGINE_PALETTE[0])
    ax.plot(x, stability["top1_agreement"].astype(float), marker="s", label="Top-1 agreement", color=ENGINE_PALETTE[1])
    ax.plot(x, stability["top2_agreement"].astype(float), marker="^", label="Top-2 agreement", color=ENGINE_PALETTE[2])
    labels = [f"{row['scenario']}\n{row['budget_pair']}" for _, row in stability.iterrows()]
    ax.set_xticks(x, labels=labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Agreement / Correlation")
    ax.set_title("Budget Stability")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")


def _plot_s2_freshness(*, ax: Any, winners: pd.DataFrame) -> None:
    s2 = winners.loc[winners["scenario"] == "s2_streaming_memory"].copy()
    if s2.empty:
        _draw_placeholder(ax=ax, message="No S2 freshness rows available")
        return
    s2 = s2.sort_values(["budget_sort", "rank_within_budget", "engine"], kind="stable").groupby("engine", as_index=False).first()
    x = np.arange(len(s2))
    width = 0.35
    ax.bar(x - width / 2, s2["freshness_hit_at_1s"].astype(float), width=width, label="hit@1s", color=ENGINE_PALETTE[0])
    ax.bar(x + width / 2, s2["freshness_hit_at_5s"].astype(float), width=width, label="hit@5s", color=ENGINE_PALETTE[1])
    ax.set_xticks(x, labels=s2["engine"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Freshness Hit")
    ax.set_title("S2 Freshness")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")


def _engine_color(engine: str) -> str:
    normalized = str(engine).strip().lower()
    index = abs(hash(normalized)) % len(ENGINE_PALETTE)
    return ENGINE_PALETTE[index]


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
            "figure.facecolor": FIGURE_FACE_COLOR,
            "savefig.facecolor": FIGURE_FACE_COLOR,
            "grid.color": GRID_COLOR,
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
