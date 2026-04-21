"""High-level report bundle export orchestration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from .plots import generate_figures, load_results
from .tables import export_tables
from maxionbench.tools.validate_outputs import validate_path

_MILESTONE_ID_RE = re.compile(r"^M[0-9]+$")
_ROBUSTNESS_SCENARIOS = {"s2_filtered_ann", "s3_churn_smooth", "s3b_churn_bursty"}


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
        if "rtt_baseline_request_profile" in message:
            raise RuntimeError(
                "Legacy run artifacts detected: pinned RTT baseline request-profile metadata is missing or invalid. "
                "Re-run benchmarks to regenerate artifacts with `rtt_baseline_request_profile` in run_metadata.json."
            ) from exc
        raise
    frame = load_results(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_policy = _report_output_policy(mode=mode, out_dir=out_dir)
    figure_paths = generate_figures(input_dir=input_dir, out_dir=out_dir, mode=mode, output_policy=output_policy)
    table_paths = export_tables(frame=frame, out_dir=out_dir, mode=mode, output_policy=output_policy)
    _enforce_t3_robustness_computable(table_paths=table_paths)
    return {"figures": figure_paths, "tables": table_paths}


def _enforce_report_output_path_policy(*, mode: str, out_dir: Path) -> None:
    if mode != "milestones":
        return
    milestone_root = _milestone_root()
    resolved_out = out_dir.resolve()
    if resolved_out == milestone_root:
        raise ValueError(
            "Milestone report output must use an explicit milestone directory `artifacts/figures/milestones/Mx` "
            "(for example `artifacts/figures/milestones/M3`)."
        )
    try:
        rel = resolved_out.relative_to(milestone_root)
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


def _report_output_policy(*, mode: str, out_dir: Path) -> dict[str, Any]:
    resolved_out = out_dir.resolve()
    policy: dict[str, Any] = {
        "mode": mode,
        "resolved_out_dir": str(resolved_out),
        "output_path_class": "final",
        "milestone_id": None,
    }
    if mode != "milestones":
        return policy

    policy["output_path_class"] = "milestones_noncanonical"
    milestone_root = _milestone_root()
    policy["milestone_root"] = str(milestone_root)
    try:
        rel = resolved_out.relative_to(milestone_root)
    except ValueError:
        return policy
    if rel.parts and _MILESTONE_ID_RE.fullmatch(rel.parts[0]):
        policy["output_path_class"] = "milestones_mx"
        policy["milestone_id"] = rel.parts[0]
    return policy


def _milestone_root() -> Path:
    return Path("artifacts/figures/milestones").resolve()


def _enforce_t3_robustness_computable(*, table_paths: list[Path]) -> None:
    t3_path = next((path for path in table_paths if path.name == "T3_robustness_summary.csv"), None)
    if t3_path is None or not t3_path.exists():
        raise RuntimeError("report export missing T3_robustness_summary.csv; cannot verify robustness inflation computability")

    frame = pd.read_csv(t3_path)
    if frame.empty:
        return

    required = {"scenario", "engine", "dataset_bundle", "p99_inflation_status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "report export missing required T3 robustness columns: "
            + ", ".join(missing)
            + "; cannot verify robustness inflation computability"
        )

    scenario_series = frame["scenario"].astype(str)
    status_series = frame["p99_inflation_status"].astype(str)
    failing_mask = scenario_series.isin(_ROBUSTNESS_SCENARIOS) & (status_series == "not_computable")
    if not bool(failing_mask.any()):
        return

    offenders = (
        frame.loc[failing_mask, ["scenario", "engine", "dataset_bundle"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["scenario", "engine", "dataset_bundle"])
    )
    offender_tokens = [
        f"{row.scenario}/{row.engine}/{row.dataset_bundle}"
        for row in offenders.itertuples(index=False)
    ]
    raise RuntimeError(
        "T3 robustness inflation is not computable for one or more robustness scenarios: "
        + ", ".join(offender_tokens)
        + ". Ensure S2 includes the 100% selectivity anchor and S3/S3b runs have matched S1 baselines."
    )
