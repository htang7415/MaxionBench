"""Regenerate paper-facing MaxionBench figures from archived run artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FIGURE_STEMS = (
    "maxionbench_decision_audit_conceptual",
    "portable_decision_surface",
    "s3_paired_audit_forest",
    "portable_task_cost_by_budget",
    "portable_budget_stability",
    "portable_s2_post_insert_retrievability",
    "portable_mvd_sensitivity",
)
FIGURE_SUFFIXES = (".pdf", ".png", ".svg", ".meta.json")
REQUIRED_SUFFIXES = FIGURE_SUFFIXES
STALE_FIGURE_STEMS = ("portable_s2_freshness",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh paper/figures from the MaxionBench report generator."
    )
    parser.add_argument("--input", default="artifacts/runs/portable")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--work-dir", default="artifacts/figures/paper_refresh")
    parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from maxionbench.reports.portable_exports import generate_portable_report_bundle

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_figure_files(directory=work_dir)
    _remove_stale_figure_files(directory=out_dir)

    generate_portable_report_bundle(
        input_dir=Path(args.input),
        out_dir=work_dir,
        conformance_matrix_path=Path(args.conformance_matrix),
        behavior_dir=Path(args.behavior_dir),
    )

    missing: list[str] = []
    copied: list[Path] = []
    for stem in FIGURE_STEMS:
        for suffix in FIGURE_SUFFIXES:
            source = work_dir / f"{stem}{suffix}"
            if source.exists():
                destination = out_dir / source.name
                shutil.copy2(source, destination)
                copied.append(destination)
            elif suffix in REQUIRED_SUFFIXES:
                missing.append(str(source))

    if missing:
        raise RuntimeError("missing expected generated figure files: " + ", ".join(missing))

    _remove_stale_figure_files(directory=out_dir)

    for path in copied:
        print(path)
    return 0


def _remove_stale_figure_files(*, directory: Path) -> None:
    for path in directory.glob("._*"):
        if path.is_file():
            path.unlink()
    for stem in STALE_FIGURE_STEMS:
        for suffix in (".pdf", ".png", ".svg", ".meta.json"):
            (directory / f"{stem}{suffix}").unlink(missing_ok=True)
            (directory / f"._{stem}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
