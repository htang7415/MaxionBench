"""Compare two result bundles cell by cell: do the 95% CIs overlap?"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from maxionbench.harness.results import ExperimentResult, from_dict

DEFAULT_METRICS = ("slo_attainment", "goodput_rps", "prefix_cache_hit_ratio", "ttft_p50_ms", "ttft_p99_ms")


def load_result(path: Path) -> ExperimentResult:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    return from_dict(ExperimentResult, json.loads(path.read_text(encoding="utf-8")))


def compare(a: ExperimentResult, b: ExperimentResult, metrics: Sequence[str] = DEFAULT_METRICS) -> dict[str, Any]:
    b_cells = {c.cell_id: c for c in b.cells}
    rows = []
    for cell in a.cells:
        other = b_cells.get(cell.cell_id)
        if other is None:
            continue
        for metric in metrics:
            x, y = cell.metrics.get(metric), other.metrics.get(metric)
            if x is None or y is None:
                continue
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "metric": metric,
                    "a": [round(x.ci_low, 4), round(x.mean, 4), round(x.ci_high, 4)],
                    "b": [round(y.ci_low, 4), round(y.mean, 4), round(y.ci_high, 4)],
                    "overlap": x.ci_low <= y.ci_high and y.ci_low <= x.ci_high,
                }
            )
    return {
        "a": a.run_id,
        "b": b.run_id,
        "same_spec": a.provenance.spec_fingerprint == b.provenance.spec_fingerprint,
        "overlapping": sum(r["overlap"] for r in rows),
        "compared": len(rows),
        "rows": rows,
    }
