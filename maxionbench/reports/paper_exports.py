"""High-level report bundle export orchestration."""

from __future__ import annotations

from pathlib import Path

from .plots import generate_figures, load_results
from .tables import export_tables
from maxionbench.tools.validate_outputs import validate_path


def generate_report_bundle(*, input_dir: Path, out_dir: Path, mode: str) -> dict[str, list[Path]]:
    if mode not in {"milestones", "final"}:
        raise ValueError("mode must be one of: milestones, final")
    resolved_input = input_dir.resolve()
    try:
        validate_path(resolved_input)
    except ValueError as exc:
        message = str(exc)
        if "missing stage timing columns" in message:
            raise RuntimeError(
                "Legacy run artifacts detected: stage timing columns are missing. "
                f"Run `maxionbench migrate-stage-timing --input {resolved_input}` and retry."
            ) from exc
        if (
            "missing resource columns" in message
            or "missing required RHU metadata mapping" in message
            or "resource_profile missing keys" in message
            or "rhu_references missing keys" in message
        ):
            raise RuntimeError(
                "Legacy run artifacts detected: RHU resource profile fields are missing. "
                "Re-run benchmarks to regenerate artifacts with resource columns and RHU metadata, then retry report generation."
            ) from exc
        raise
    frame = load_results(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_figures(input_dir=input_dir, out_dir=out_dir, mode=mode)
    table_paths = export_tables(frame=frame, out_dir=out_dir, mode=mode)
    return {"figures": figure_paths, "tables": table_paths}
