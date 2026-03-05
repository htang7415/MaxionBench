"""Verify engine readiness from conformance outputs and behavior-card coverage."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import pandas as pd

from maxionbench.tools.verify_behavior_cards import verify_behavior_cards

REQUIRED_ADAPTERS = (
    "qdrant",
    "milvus",
    "weaviate",
    "opensearch",
    "pgvector",
    "lancedb-service",
    "lancedb-inproc",
    "faiss-cpu",
    "faiss-gpu",
)
BEHAVIOR_CARD_BY_ADAPTER = {
    "qdrant": "qdrant.md",
    "milvus": "milvus.md",
    "weaviate": "weaviate.md",
    "opensearch": "opensearch.md",
    "pgvector": "pgvector.md",
    "lancedb-service": "lancedb.md",
    "lancedb-inproc": "lancedb.md",
    "faiss-cpu": "faiss_cpu.md",
    "faiss-gpu": "faiss_gpu.md",
}


def verify_engine_readiness(
    *,
    conformance_matrix_path: Path,
    behavior_dir: Path,
    allow_gpu_unavailable: bool = False,
    allow_nonpass_status: bool = False,
) -> dict[str, Any]:
    matrix_path = conformance_matrix_path.resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Conformance matrix not found: {matrix_path}")

    frame = pd.read_csv(matrix_path)
    if not {"adapter", "status"}.issubset(frame.columns):
        raise ValueError("Conformance matrix must include `adapter` and `status` columns")

    behavior_summary = verify_behavior_cards(behavior_dir)
    errors: list[dict[str, Any]] = []
    if not behavior_summary["pass"]:
        for item in behavior_summary["errors"]:
            payload = dict(item)
            payload["source"] = "behavior_cards"
            errors.append(payload)

    required_adapters = list(REQUIRED_ADAPTERS)
    if allow_gpu_unavailable:
        required_adapters = [name for name in required_adapters if name != "faiss-gpu"]

    for adapter in required_adapters:
        matches = frame[frame["adapter"] == adapter]
        if matches.empty:
            errors.append(
                {
                    "source": "conformance_matrix",
                    "adapter": adapter,
                    "field": "status",
                    "expected": "pass row present",
                    "actual": "missing",
                    "message": f"missing conformance row for adapter `{adapter}`",
                }
            )
            continue
        statuses = sorted({str(v) for v in matches["status"].tolist()})
        if (not allow_nonpass_status) and ("pass" not in statuses):
            errors.append(
                {
                    "source": "conformance_matrix",
                    "adapter": adapter,
                    "field": "status",
                    "expected": "pass",
                    "actual": statuses,
                    "message": f"adapter `{adapter}` has no pass status in conformance matrix",
                }
            )
        expected_card = BEHAVIOR_CARD_BY_ADAPTER.get(adapter)
        if expected_card:
            card_path = behavior_dir.resolve() / expected_card
            if not card_path.exists():
                errors.append(
                    {
                        "source": "behavior_cards",
                        "adapter": adapter,
                        "field": "card_file",
                        "expected": expected_card,
                        "actual": "missing",
                        "message": f"adapter `{adapter}` missing behavior card file `{expected_card}`",
                    }
                )

    status_counts: dict[str, int] = {}
    for value in frame["status"].tolist():
        key = str(value)
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "conformance_matrix_path": str(matrix_path),
        "behavior_dir": str(behavior_dir.resolve()),
        "allow_gpu_unavailable": allow_gpu_unavailable,
        "allow_nonpass_status": allow_nonpass_status,
        "required_adapters": required_adapters,
        "conformance_rows": int(len(frame)),
        "conformance_status_counts": dict(sorted(status_counts.items())),
        "behavior_cards_ok": bool(behavior_summary["pass"]),
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify conformance + behavior-card readiness before benchmarking")
    parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    parser.add_argument(
        "--allow-gpu-unavailable",
        action="store_true",
        help="Allow missing/non-pass `faiss-gpu` readiness when GPU Track B is intentionally omitted.",
    )
    parser.add_argument(
        "--allow-nonpass-status",
        action="store_true",
        help="Allow readiness checks to pass when adapter rows exist but have non-pass statuses.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_engine_readiness(
        conformance_matrix_path=Path(args.conformance_matrix),
        behavior_dir=Path(args.behavior_dir),
        allow_gpu_unavailable=bool(args.allow_gpu_unavailable),
        allow_nonpass_status=bool(args.allow_nonpass_status),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print("engine readiness verification passed")
        else:
            print(f"engine readiness verification failed: {summary['error_count']} issue(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
