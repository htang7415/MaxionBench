"""High-level report bundle export orchestration."""

from __future__ import annotations

from pathlib import Path
import re

from .plots import generate_figures, load_results
from .tables import export_tables
from maxionbench.tools.validate_outputs import validate_path

_MILESTONE_ROOT = Path("artifacts/figures/milestones").resolve()
_MILESTONE_ID_RE = re.compile(r"^M[0-9]+$")


def generate_report_bundle(*, input_dir: Path, out_dir: Path, mode: str) -> dict[str, list[Path]]:
    if mode not in {"milestones", "final"}:
        raise ValueError("mode must be one of: milestones, final")
    _enforce_report_output_path_policy(mode=mode, out_dir=out_dir)
    resolved_input = input_dir.resolve()
    try:
        validate_path(resolved_input, strict_schema=True)
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
        if "missing ground truth metadata keys" in message or "ground_truth_" in message:
            raise RuntimeError(
                "Legacy run artifacts detected: ground truth metadata fields are missing or invalid. "
                "Re-run benchmarks to regenerate artifacts with complete ground-truth provenance in run_metadata.json."
            ) from exc
        if "hardware/runtime summary" in message or "hardware_runtime" in message:
            raise RuntimeError(
                "Legacy run artifacts detected: hardware/runtime summary metadata is missing or invalid. "
                "Re-run benchmarks to regenerate artifacts with `hardware_runtime` fields in run_metadata.json."
            ) from exc
        raise
    frame = load_results(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_figures(input_dir=input_dir, out_dir=out_dir, mode=mode)
    table_paths = export_tables(frame=frame, out_dir=out_dir, mode=mode)
    return {"figures": figure_paths, "tables": table_paths}


def _enforce_report_output_path_policy(*, mode: str, out_dir: Path) -> None:
    if mode != "milestones":
        return
    resolved_out = out_dir.resolve()
    if resolved_out == _MILESTONE_ROOT:
        raise ValueError(
            "Milestone report output must use an explicit milestone directory `artifacts/figures/milestones/Mx` "
            "(for example `artifacts/figures/milestones/M3`)."
        )
    try:
        rel = resolved_out.relative_to(_MILESTONE_ROOT)
    except ValueError:
        return
    if not rel.parts:
        raise ValueError(
            "Milestone report output must use an explicit milestone directory `artifacts/figures/milestones/Mx`."
        )
    if not _MILESTONE_ID_RE.fullmatch(rel.parts[0]):
        raise ValueError(
            "Milestone report output under `artifacts/figures/milestones/` must start with an `Mx` directory "
            "(for example `artifacts/figures/milestones/M2`)."
        )
