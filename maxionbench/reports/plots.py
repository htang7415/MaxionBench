"""Shared report loading and style constants for portable reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from maxionbench.schemas.result_schema import RUN_STATUS_FILENAME, read_run_status

FONT_SIZE = 9
PANEL_WIDTH_IN = 3.45
PANEL_HEIGHT_IN = 2.25
PANEL_PX = 1035
DPI = 300
STYLE_VERSION = "portable_v4_neurips_halfwidth"
FIGURE_FACE_COLOR = "#ffffff"
TEXT_COLOR = "#1f2933"
GRID_COLOR = "#d9dee5"
ENGINE_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#6F4E37",
    "#F0E442",
)


def load_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("results.parquet"))
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        run_dir = path.parent
        status_path = run_dir / RUN_STATUS_FILENAME
        if status_path.exists():
            payload = read_run_status(status_path)
            status = str(payload.get("status") or "").strip().lower()
            if status != "success":
                continue
        frame = pd.read_parquet(path)
        frame["__run_path"] = str(run_dir)
        metadata = _load_run_metadata(run_dir)
        frame["__meta_run_id"] = str(metadata.get("run_id") or "")
        frame["__meta_config_fingerprint"] = str(metadata.get("config_fingerprint") or "")
        frame["__meta_ground_truth_source"] = str(metadata.get("ground_truth_source") or "")
        frame["__meta_ground_truth_metric"] = str(metadata.get("ground_truth_metric") or "")
        frame["__meta_ground_truth_engine"] = str(metadata.get("ground_truth_engine") or "")
        frame["__meta_ground_truth_k"] = _safe_float(metadata.get("ground_truth_k"))
        frame["__meta_profile"] = str(metadata.get("profile") or "")
        frame["__meta_budget_level"] = str(metadata.get("budget_level") or "")
        frame["__meta_embedding_model"] = str(metadata.get("embedding_model") or "")
        frame["__meta_embedding_dim"] = _safe_float(metadata.get("embedding_dim"))
        frame["__meta_c_llm_in"] = _safe_float(metadata.get("c_llm_in"))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["scenario", "engine", "repeat_idx", "clients_read", "quality_target"] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return merged


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
