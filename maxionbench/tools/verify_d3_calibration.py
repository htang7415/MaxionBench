"""Verify D3 calibration YAML is paper-ready for D3 robustness scenarios."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import yaml

from maxionbench.datasets.d3_calibrate import PAPER_MIN_CALIBRATION_VECTORS, paper_calibration_issues


def verify_d3_calibration_file(
    *,
    path: Path,
    min_vectors: int = PAPER_MIN_CALIBRATION_VECTORS,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"d3 params file does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"d3 params file is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("d3 params file root must be a mapping")

    issues = paper_calibration_issues(payload=payload, min_vectors=int(min_vectors))
    return {
        "path": str(resolved),
        "min_vectors": int(min_vectors),
        "paper_ready": len(issues) == 0,
        "error_count": len(issues),
        "errors": issues,
        "k_clusters": payload.get("k_clusters"),
        "calibration_vector_count": payload.get("calibration_vector_count"),
        "calibration_source": payload.get("calibration_source"),
        "calibration_paper_ready_flag": payload.get("calibration_paper_ready"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify D3 calibration YAML for paper-ready thresholds")
    parser.add_argument("--d3-params", default="artifacts/calibration/d3_params.yaml")
    parser.add_argument("--min-vectors", type=int, default=PAPER_MIN_CALIBRATION_VECTORS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_d3_calibration_file(
        path=Path(args.d3_params),
        min_vectors=int(args.min_vectors),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["paper_ready"]:
            print(f"d3 calibration verification passed: {summary['path']}")
        else:
            print(f"d3 calibration verification failed: {summary['error_count']} issue(s)")
            for message in summary["errors"]:
                print(f"- {message}")
    if args.strict and not bool(summary["paper_ready"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
