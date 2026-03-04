"""High-level report bundle export orchestration."""

from __future__ import annotations

from pathlib import Path

from .plots import generate_figures, load_results
from .tables import export_tables


def generate_report_bundle(*, input_dir: Path, out_dir: Path, mode: str) -> dict[str, list[Path]]:
    if mode not in {"milestones", "final"}:
        raise ValueError("mode must be one of: milestones, final")
    frame = load_results(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_figures(input_dir=input_dir, out_dir=out_dir, mode=mode)
    table_paths = export_tables(frame=frame, out_dir=out_dir, mode=mode)
    return {"figures": figure_paths, "tables": table_paths}
