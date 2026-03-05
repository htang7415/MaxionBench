"""Deterministic figure generation for milestone/final report bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "maxionbench-mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib.pyplot as plt

FONT_SIZE = 16
PANEL_PX = 600
DPI = 100


@dataclass(frozen=True)
class FigureSpec:
    name: str
    scenario_hint: str | None = None


MILESTONE_SPECS = [
    FigureSpec("m1_conformance_coverage_matrix"),
    FigureSpec("m1_semantics_behavior_matrix"),
    FigureSpec("m2_runner_stage_timing"),
    FigureSpec("m2_schema_validation_summary"),
    FigureSpec("m3_s1d1_pareto_qdrant", "s1_ann_frontier"),
    FigureSpec("m3_s1d1_latency_clients_qdrant", "s1_ann_frontier"),
    FigureSpec("m3_s1d1_build_load_footprint_qdrant", "s1_ann_frontier"),
    FigureSpec("m4_s1_comparison_qdrant_pgvector", "s1_ann_frontier"),
    FigureSpec("m4_s1_cost_quality_qdrant_pgvector", "s1_ann_frontier"),
    FigureSpec("m5_s1_all_engines_pareto_d1", "s1_ann_frontier"),
    FigureSpec("m5_s1_all_engines_pareto_d2", "s1_ann_frontier"),
    FigureSpec("m5_time_to_ready_and_footprint_all_engines"),
    FigureSpec("m5_runtime_rtt_baseline_by_engine"),
    FigureSpec("m6_d3_testA_concentration_distribution", "calibrate_d3"),
    FigureSpec("m6_d3_testB_cluster_spread_at_1pct", "calibrate_d3"),
    FigureSpec("m6_s2_selectivity_curves_pre_post_calibration", "s2_filtered_ann"),
    FigureSpec("m7_s3_smooth_churn_latency_and_sla", "s3_churn_smooth"),
    FigureSpec("m7_s3b_bursty_churn_latency_and_sla", "s3b_churn_bursty"),
    FigureSpec("m7_s4_hybrid_dense_vs_bm25_dense", "s4_hybrid"),
    FigureSpec("m7_s5_rerank_utility_latency_tradeoff", "s5_rerank"),
    FigureSpec("m8_s6a_multigranularity_fusion_gain", "s6_fusion"),
    FigureSpec("m8_s6b_dense_bm25_fusion_gain", "s6_fusion"),
]

FINAL_SPECS = [
    FigureSpec("F1_s1_pareto_frontiers", "s1_ann_frontier"),
    FigureSpec("F2_s2_selectivity_vs_p99_inflation_and_recall", "s2_filtered_ann"),
    FigureSpec("F3_s3_s3b_robustness_under_churn"),
    FigureSpec("F4_s4_s5_utility_overhead_tradeoff"),
    FigureSpec("F5_s6_fusion_results", "s6_fusion"),
    FigureSpec("A1_runtime_rtt_baseline"),
    FigureSpec("A2_build_load_time_and_footprint"),
    FigureSpec("A3_repeatability_variance_ci"),
]


def generate_figures(*, input_dir: Path, out_dir: Path, mode: str) -> list[Path]:
    frame = load_results(input_dir)
    conformance = _load_conformance_matrix(input_dir)
    behavior = _load_behavior_matrix()
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    specs = MILESTONE_SPECS if mode == "milestones" else FINAL_SPECS
    generated: list[Path] = []
    has_s6_data = bool("scenario" in frame.columns and (frame["scenario"] == "s6_fusion").any())
    for spec in specs:
        if not has_s6_data and mode == "milestones" and spec.name.startswith("m8_s6"):
            continue
        if not has_s6_data and mode == "final" and spec.name == "F5_s6_fusion_results":
            continue
        subset = _subset(frame, scenario_hint=spec.scenario_hint)
        fig, ax = plt.subplots(figsize=(PANEL_PX / DPI, PANEL_PX / DPI), dpi=DPI)
        _render_plot(ax=ax, frame=subset, title=spec.name, conformance=conformance, behavior=behavior)
        out_png = out_dir / f"{spec.name}.png"
        fig.savefig(out_png, dpi=DPI, format="png")
        plt.close(fig)

        run_ids = _unique_str_values(subset, "run_id")
        config_fingerprints = _unique_str_values(subset, "__meta_config_fingerprint")
        dataset_bundles = _unique_str_values(subset, "dataset_bundle")
        seeds = _unique_int_values(subset, "seed")
        meta = {
            "figure_name": spec.name,
            "mode": mode,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "rows_total": int(len(frame)),
            "rows_used": int(len(subset)),
            "conformance_rows": int(len(conformance)),
            "engines": sorted({str(v) for v in subset.get("engine", pd.Series(dtype=str)).tolist()}),
            "scenarios": sorted({str(v) for v in subset.get("scenario", pd.Series(dtype=str)).tolist()}),
            "dataset_bundles": dataset_bundles,
            "seeds": seeds,
            "run_ids": run_ids,
            "config_fingerprints": config_fingerprints,
            "font_size": FONT_SIZE,
            "panel_pixels": PANEL_PX,
            "dpi": DPI,
        }
        _write_meta(out_png, meta)
        generated.append(out_png)
    if not has_s6_data:
        generated.extend(_write_s6_deferred_note(mode=mode, out_dir=out_dir, frame=frame))
    return generated


def load_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("results.parquet"))
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        run_dir = path.parent
        frame["__run_path"] = str(run_dir)
        metadata = _load_run_metadata(run_dir)
        frame["__meta_run_id"] = str(metadata.get("run_id") or "")
        frame["__meta_config_fingerprint"] = str(metadata.get("config_fingerprint") or "")
        frame["__meta_ground_truth_source"] = str(metadata.get("ground_truth_source") or "")
        frame["__meta_ground_truth_metric"] = str(metadata.get("ground_truth_metric") or "")
        frame["__meta_ground_truth_engine"] = str(metadata.get("ground_truth_engine") or "")
        frame["__meta_ground_truth_k"] = _safe_float(metadata.get("ground_truth_k"))
        rhu_weights = metadata.get("rhu_weights")
        if not isinstance(rhu_weights, dict):
            rhu_weights = {}
        rhu_refs = metadata.get("rhu_references")
        if not isinstance(rhu_refs, dict):
            rhu_refs = {}
        resource_profile = metadata.get("resource_profile")
        if not isinstance(resource_profile, dict):
            resource_profile = {}
        frame["__meta_w_c"] = _safe_float(rhu_weights.get("w_c"))
        frame["__meta_w_g"] = _safe_float(rhu_weights.get("w_g"))
        frame["__meta_w_r"] = _safe_float(rhu_weights.get("w_r"))
        frame["__meta_w_d"] = _safe_float(rhu_weights.get("w_d"))
        frame["__meta_c_ref_vcpu"] = _safe_float(rhu_refs.get("c_ref_vcpu"))
        frame["__meta_g_ref_gpu"] = _safe_float(rhu_refs.get("g_ref_gpu"))
        frame["__meta_r_ref_gib"] = _safe_float(rhu_refs.get("r_ref_gib"))
        frame["__meta_d_ref_tb"] = _safe_float(rhu_refs.get("d_ref_tb"))
        frame["__meta_profile_cpu_vcpu"] = _safe_float(resource_profile.get("cpu_vcpu"))
        frame["__meta_profile_gpu_count"] = _safe_float(resource_profile.get("gpu_count"))
        frame["__meta_profile_ram_gib"] = _safe_float(resource_profile.get("ram_gib"))
        frame["__meta_profile_disk_tb"] = _safe_float(resource_profile.get("disk_tb"))
        frame["__meta_profile_rhu_rate"] = _safe_float(resource_profile.get("rhu_rate"))
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["scenario", "engine", "repeat_idx", "clients_read", "quality_target"] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return merged


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )


def _subset(frame: pd.DataFrame, *, scenario_hint: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not scenario_hint:
        return frame
    if "scenario" not in frame.columns:
        return frame.iloc[0:0]
    filtered = frame[frame["scenario"] == scenario_hint]
    return filtered if not filtered.empty else frame.iloc[0:0]


def _render_plot(
    *,
    ax: Any,
    frame: pd.DataFrame,
    title: str,
    conformance: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    ax.set_title(title)
    if title == "m1_conformance_coverage_matrix":
        _render_conformance_matrix(ax=ax, conformance=conformance)
        return
    if title == "m1_semantics_behavior_matrix":
        _render_behavior_matrix(ax=ax, behavior=behavior)
        return
    if title == "m2_runner_stage_timing":
        _render_stage_timing(ax=ax, frame=frame)
        return
    if title == "m2_schema_validation_summary":
        _render_schema_validation(ax=ax, frame=frame)
        return
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if {"recall_at_10", "p99_ms"}.issubset(frame.columns):
        for engine, sub in frame.groupby("engine", sort=True):
            ax.scatter(sub["recall_at_10"], sub["p99_ms"], label=str(engine), s=36)
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("p99 (ms)")
    elif {"clients_read", "p99_ms"}.issubset(frame.columns):
        for engine, sub in frame.groupby("engine", sort=True):
            sub2 = sub.sort_values("clients_read", kind="stable")
            ax.plot(sub2["clients_read"], sub2["p99_ms"], marker="o", label=str(engine))
        ax.set_xlabel("Clients")
        ax.set_ylabel("p99 (ms)")
    else:
        counts = frame.groupby("scenario", sort=True).size()
        ax.bar(counts.index.tolist(), counts.values.tolist())
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=30)

    if frame["engine"].nunique() <= 8:
        ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)


def _render_conformance_matrix(*, ax: Any, conformance: pd.DataFrame) -> None:
    if conformance.empty:
        ax.text(0.5, 0.5, "No conformance data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    adapters = sorted({str(v) for v in conformance["adapter"].tolist()})
    operations = [
        "healthcheck",
        "flush_visibility",
        "filter",
        "update_vector",
        "update_payload",
        "delete",
        "batch_query",
        "stats",
        "optimize",
    ]
    matrix = np.zeros((len(operations), len(adapters)), dtype=np.float32)
    status_by_adapter = {str(row["adapter"]): str(row["status"]) for _, row in conformance.iterrows()}
    for j, adapter in enumerate(adapters):
        passed = status_by_adapter.get(adapter) == "pass"
        matrix[:, j] = 1.0 if passed else 0.0
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(adapters)))
    ax.set_xticklabels(adapters, rotation=30, ha="right")
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels(operations)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _render_behavior_matrix(*, ax: Any, behavior: pd.DataFrame) -> None:
    if behavior.empty:
        ax.text(0.5, 0.5, "No behavior docs", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    engines = behavior["engine"].tolist()
    columns = [col for col in behavior.columns if col != "engine"]
    matrix = behavior[columns].to_numpy(dtype=np.float32).T
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, rotation=30, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _render_stage_timing(*, ax: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No run data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    setup_s = float(frame.get("setup_elapsed_s", pd.Series([0.0])).median())
    warmup_s = float(frame.get("warmup_elapsed_s", pd.Series([0.0])).median())
    measure_s = float(frame.get("measure_elapsed_s", pd.Series([0.0])).median())
    export_s = float(frame.get("export_elapsed_s", pd.Series([0.0])).median())
    labels = ["setup", "warmup", "measure", "export"]
    values = [setup_s, warmup_s, measure_s, export_s]
    ax.bar(labels, values)
    ax.set_ylabel("Seconds (median)")
    ax.grid(alpha=0.2, axis="y")


def _render_schema_validation(*, ax: Any, frame: pd.DataFrame) -> None:
    if frame.empty or "__run_path" not in frame.columns:
        ax.text(0.5, 0.5, "No run paths", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    from maxionbench.tools.validate_outputs import validate_run_directory

    run_paths = sorted({str(path) for path in frame["__run_path"].tolist() if str(path)})
    success = 0
    failed = 0
    for run_path in run_paths:
        try:
            validate_run_directory(Path(run_path))
            success += 1
        except Exception:
            failed += 1
    ax.bar(["valid", "invalid"], [success, failed], color=["#4daf4a", "#e41a1c"])
    ax.set_ylabel("Run Count")
    ax.grid(alpha=0.2, axis="y")


def _load_conformance_matrix(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "conformance_matrix.csv",
        input_dir.parent / "conformance" / "conformance_matrix.csv",
        Path("artifacts/conformance/conformance_matrix.csv").resolve(),
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if {"adapter", "status"}.issubset(frame.columns):
                return frame
    return pd.DataFrame(columns=["adapter", "status"])


def _load_behavior_matrix() -> pd.DataFrame:
    behavior_dir = Path("docs/behavior").resolve()
    if not behavior_dir.exists():
        return pd.DataFrame(columns=["engine", "flush", "delete", "update", "compaction", "persistence"])
    rows: list[dict[str, Any]] = []
    for path in sorted(behavior_dir.glob("*.md")):
        if path.name == "_template.md":
            continue
        text = path.read_text(encoding="utf-8").lower()
        rows.append(
            {
                "engine": path.stem,
                "flush": 1.0 if "flush" in text or "commit" in text else 0.0,
                "delete": 1.0 if "delete" in text else 0.0,
                "update": 1.0 if "update" in text else 0.0,
                "compaction": 1.0 if "compaction" in text or "optimize" in text else 0.0,
                "persistence": 1.0 if "persist" in text or "durable" in text else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["engine", "flush", "delete", "update", "compaction", "persistence"])
    frame = pd.DataFrame(rows)
    return frame.sort_values("engine", kind="stable").reset_index(drop=True)


def _write_meta(png_path: Path, payload: Mapping[str, Any]) -> None:
    meta_path = png_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _unique_str_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = {str(v) for v in frame[column].tolist() if str(v)}
    return sorted(values)


def _unique_int_values(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return []
    values: set[int] = set()
    for value in frame[column].tolist():
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        try:
            values.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(values)


def _write_s6_deferred_note(*, mode: str, out_dir: Path, frame: pd.DataFrame) -> list[Path]:
    name = "m8_deferred_note" if mode == "milestones" else "F5_deferred_note"
    note_path = out_dir / f"{name}.md"
    meta_path = out_dir / f"{name}.meta.json"
    run_ids = _unique_str_values(frame, "run_id")
    config_fingerprints = _unique_str_values(frame, "__meta_config_fingerprint")
    dataset_bundles = _unique_str_values(frame, "dataset_bundle")
    seeds = _unique_int_values(frame, "seed")
    note_path.write_text(
        "S6 was deferred in this report bundle because no `s6_fusion` scenario rows were found.\n",
        encoding="utf-8",
    )
    payload = {
        "name": name,
        "mode": mode,
        "deferred": True,
        "reason": "no_s6_rows",
        "engines": _unique_str_values(frame, "engine"),
        "scenarios": _unique_str_values(frame, "scenario"),
        "dataset_bundles": dataset_bundles,
        "seeds": seeds,
        "run_ids": run_ids,
        "config_fingerprints": config_fingerprints,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return [note_path, meta_path]
