"""Summarize the larger same-machine S2 FAISS/Qdrant comparison."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("artifacts/runs/neurips_rerun/s2_larger_same_machine_b2")
DEFAULT_TEMPLATE = "s2_streaming_memory__bge-small-en-v1-5"
DEFAULT_OUT = Path("paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json")

ENGINE_DIRS = {
    "faiss-cpu": "faiss_cpu",
    "qdrant": "qdrant",
}


def _read_jsonl(path: Path, *, repeat_idx: int, engine: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["repeat_idx"] = int(repeat_idx)
            row["engine"] = engine
            rows.append(row)
    return rows


def _observation_paths(frame: pd.DataFrame, repo_root: Path) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for row in frame.to_dict(orient="records"):
        params = json.loads(str(row.get("search_params_json") or "{}"))
        observation_path = params.get("observation_path")
        if not observation_path:
            continue
        path = Path(str(observation_path))
        if not path.is_absolute():
            path = repo_root / path
        paths.append((int(row.get("repeat_idx", 0)), path))
    return paths


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _ci95(values: np.ndarray) -> list[float] | None:
    if values.size == 0:
        return None
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(42)
    draws = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _paired_delta(records: list[dict[str, Any]], metric: str, *, keys: list[str]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(records)
    if frame.empty or metric not in frame:
        return {"paired_count": 0}
    pivot = frame.pivot_table(index=keys, columns="engine", values=metric, aggfunc="first")
    if "faiss-cpu" not in pivot or "qdrant" not in pivot:
        return {"paired_count": 0}
    paired = pivot[["faiss-cpu", "qdrant"]].dropna()
    deltas = (paired["qdrant"] - paired["faiss-cpu"]).to_numpy(dtype=np.float64)
    return {
        "paired_count": int(deltas.size),
        "mean_delta_qdrant_minus_faiss": float(np.mean(deltas)) if deltas.size else None,
        "median_delta": float(np.median(deltas)) if deltas.size else None,
        "ci95": _ci95(deltas),
        "probability_delta_positive": float(np.mean(deltas > 0.0)) if deltas.size else None,
        "zero_delta_fraction": float(np.mean(np.isclose(deltas, 0.0))) if deltas.size else None,
        "changed_events": int(np.sum(~np.isclose(deltas, 0.0))) if deltas.size else None,
    }


def summarize(*, output_root: Path, template: str, repo_root: Path) -> dict[str, Any]:
    root = output_root / template
    summary: dict[str, Any] = {
        "name": "s2_larger_same_machine_faiss_qdrant",
        "output_root": str(output_root),
        "template": template,
        "status": "incomplete",
        "protocol": {
            "budget_level": "b2",
            "scenario": "s2_streaming_memory",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "engines": ["faiss-cpu", "qdrant"],
            "clients_read": 8,
            "clients_write": 2,
            "repeats": 2,
            "search_sweep": [{"hnsw_ef": 64}],
            "phase_max_requests_per_phase": 1000,
            "s2_max_freshness_events": 100,
            "machine_scope": "same local machine as the rest of the project",
        },
        "aggregate": {},
        "runs": [],
        "paired_quality": {},
        "paired_freshness": {},
        "paired_latency": {},
        "validation": {
            "error_count": 0,
            "errors": [],
        },
    }

    all_quality: list[dict[str, Any]] = []
    all_freshness: list[dict[str, Any]] = []

    for engine, dirname in ENGINE_DIRS.items():
        engine_dir = root / dirname
        results_path = engine_dir / "results.parquet"
        status_path = engine_dir / "run_status.json"
        if not results_path.exists():
            summary["validation"]["errors"].append(f"missing results: {results_path}")
            continue
        frame = pd.read_parquet(results_path)
        rows = frame.to_dict(orient="records")
        summary["runs"].extend(rows)
        errors = int(pd.to_numeric(frame.get("errors", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if str(status.get("status", "")).lower() != "success":
                summary["validation"]["errors"].append(f"{engine} status is not success: {status}")
        else:
            summary["validation"]["errors"].append(f"missing run status: {status_path}")

        observations: list[dict[str, Any]] = []
        for repeat_idx, path in _observation_paths(frame, repo_root):
            if not path.exists():
                summary["validation"]["errors"].append(f"missing observation log: {path}")
                continue
            observations.extend(_read_jsonl(path, repeat_idx=repeat_idx, engine=engine))
        quality = [row for row in observations if row.get("observation_type") == "quality"]
        freshness = [row for row in observations if row.get("observation_type") == "freshness"]
        all_quality.extend(quality)
        all_freshness.extend(freshness)

        p99_values = [float(value) for value in frame["p99_ms"].dropna().tolist()]
        p95_values = [float(value) for value in frame["p95_ms"].dropna().tolist()]
        ndcg_values = [float(value) for value in frame["ndcg_at_10"].dropna().tolist()]
        summary["aggregate"][engine] = {
            "rows": int(len(frame)),
            "repeats": sorted(int(value) for value in frame["repeat_idx"].dropna().unique().tolist()),
            "errors_sum": errors,
            "quality_observations": len(quality),
            "freshness_observations": len(freshness),
            "ndcg_at_10_mean": _safe_mean(ndcg_values),
            "ndcg_at_10_std": _safe_std(ndcg_values),
            "recall_at_10_mean": _safe_mean([float(value) for value in frame["recall_at_10"].dropna().tolist()]),
            "mrr_at_10_mean": _safe_mean([float(value) for value in frame["mrr_at_10"].dropna().tolist()]),
            "freshness_hit_at_1s_mean": _safe_mean([float(value) for value in frame["freshness_hit_at_1s"].dropna().tolist()]),
            "freshness_hit_at_5s_mean": _safe_mean([float(value) for value in frame["freshness_hit_at_5s"].dropna().tolist()]),
            "p95_ms_mean": _safe_mean(p95_values),
            "p99_ms_mean": _safe_mean(p99_values),
            "p99_ms_min": min(p99_values) if p99_values else None,
            "p99_ms_max": max(p99_values) if p99_values else None,
            "task_cost_est_mean": _safe_mean([float(value) for value in frame["task_cost_est"].dropna().tolist()]),
            "result_path": str(results_path),
        }

    summary["validation"]["error_count"] = len(summary["validation"]["errors"])
    expected_engines = {"faiss-cpu", "qdrant"}
    observed_engines = set(summary["aggregate"].keys())
    if observed_engines == expected_engines and summary["validation"]["error_count"] == 0:
        if all(item["errors_sum"] == 0 and item["rows"] == 2 for item in summary["aggregate"].values()):
            summary["status"] = "completed"
        else:
            summary["status"] = "completed_with_metric_warnings"

    if all_quality:
        quality_keys = ["repeat_idx", "query_id"]
        for metric in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]:
            summary["paired_quality"][metric] = _paired_delta(all_quality, metric, keys=quality_keys)
        summary["paired_latency"]["latency_ms"] = _paired_delta(all_quality, "latency_ms", keys=quality_keys)
    if all_freshness:
        freshness_keys = ["repeat_idx", "event_index"]
        for metric in ["freshness_hit_at_1s", "freshness_hit_at_5s", "visibility_latency_ms"]:
            summary["paired_freshness"][metric] = _paired_delta(all_freshness, metric, keys=freshness_keys)

    summary["interpretation"] = [
        "This larger bounded run strengthens the same-machine S2 FAISS/Qdrant repeatability evidence beyond the deadline mini-bundle.",
        "Use the paired deltas to support the story only if the confidence intervals remain near zero and include zero for quality/freshness.",
        "Because freshness is still capped, frame the run as larger bounded evidence rather than a fully uncapped S2 replacement.",
    ]
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    summary = _json_safe(summarize(output_root=Path(args.output_root), template=str(args.template), repo_root=repo_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0 if summary["validation"]["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
