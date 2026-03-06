"""Verify engine readiness from conformance outputs and behavior-card coverage."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
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
    require_mock_pass: bool = False,
) -> dict[str, Any]:
    matrix_path = conformance_matrix_path.resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Conformance matrix not found: {matrix_path}")

    frame = pd.read_csv(matrix_path)
    if not {"adapter", "status"}.issubset(frame.columns):
        raise ValueError("Conformance matrix must include `adapter` and `status` columns")

    adapter_series = frame["adapter"].fillna("").astype(str).str.strip()
    status_series = frame["status"].fillna("").astype(str).str.strip()
    empty_adapter_rows = int((adapter_series == "").sum())
    empty_status_rows = int((status_series == "").sum())
    normalized = frame.copy()
    normalized["adapter"] = adapter_series
    normalized["status"] = status_series

    behavior_summary = verify_behavior_cards(behavior_dir)
    errors: list[dict[str, Any]] = []
    if empty_adapter_rows > 0:
        errors.append(
            {
                "source": "conformance_matrix",
                "field": "adapter",
                "expected": "non-empty adapter values",
                "actual": {"empty_rows": empty_adapter_rows},
                "message": f"conformance matrix contains {empty_adapter_rows} row(s) with empty adapter values",
            }
        )
    if empty_status_rows > 0:
        errors.append(
            {
                "source": "conformance_matrix",
                "field": "status",
                "expected": "non-empty status values",
                "actual": {"empty_rows": empty_status_rows},
                "message": f"conformance matrix contains {empty_status_rows} row(s) with empty status values",
            }
        )
    if not behavior_summary["pass"]:
        for item in behavior_summary["errors"]:
            payload = dict(item)
            payload["source"] = "behavior_cards"
            errors.append(payload)

    required_adapters = list(REQUIRED_ADAPTERS)
    if allow_gpu_unavailable:
        required_adapters = [name for name in required_adapters if name != "faiss-gpu"]

    for adapter in required_adapters:
        matches = normalized[normalized["adapter"] == adapter]
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
        if not allow_nonpass_status:
            if "pass" not in statuses:
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
            non_pass_statuses = sorted(status for status in statuses if status != "pass")
            if non_pass_statuses:
                errors.append(
                    {
                        "source": "conformance_matrix",
                        "adapter": adapter,
                        "field": "status",
                        "expected": ["pass"],
                        "actual": statuses,
                        "message": (
                            f"adapter `{adapter}` includes non-pass statuses in strict mode: {non_pass_statuses}"
                        ),
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

    if not allow_nonpass_status:
        strict_frame = normalized
        if allow_gpu_unavailable:
            strict_frame = strict_frame[strict_frame["adapter"] != "faiss-gpu"]
        non_pass = strict_frame[strict_frame["status"] != "pass"]
        if not non_pass.empty:
            status_counts: dict[str, int] = {}
            for value in non_pass["status"].tolist():
                key = str(value)
                status_counts[key] = status_counts.get(key, 0) + 1
            strict_scope = "excluding `faiss-gpu` rows" if allow_gpu_unavailable else "across all rows"
            errors.append(
                {
                    "source": "conformance_matrix",
                    "field": "status",
                    "expected": "all strict-scope rows == pass",
                    "actual": dict(sorted(status_counts.items())),
                    "message": (
                        f"strict readiness requires pass-only statuses {strict_scope}; "
                        f"found non-pass rows: {dict(sorted(status_counts.items()))}"
                    ),
                }
            )

    if require_mock_pass:
        mock_matches = normalized[normalized["adapter"] == "mock"]
        if mock_matches.empty:
            errors.append(
                {
                    "source": "conformance_matrix",
                    "adapter": "mock",
                    "field": "status",
                    "expected": "pass row present",
                    "actual": "missing",
                    "message": "missing conformance row for adapter `mock`",
                }
            )
        else:
            mock_statuses = sorted({str(v) for v in mock_matches["status"].tolist()})
            if "pass" not in mock_statuses:
                errors.append(
                    {
                        "source": "conformance_matrix",
                        "adapter": "mock",
                        "field": "status",
                        "expected": "pass",
                        "actual": mock_statuses,
                        "message": "adapter `mock` has no pass status in conformance matrix",
                    }
                )

    status_counts: dict[str, int] = {}
    for value in normalized["status"].tolist():
        key = str(value)
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "conformance_matrix_path": str(matrix_path),
        "behavior_dir": str(behavior_dir.resolve()),
        "allow_gpu_unavailable": allow_gpu_unavailable,
        "allow_nonpass_status": allow_nonpass_status,
        "require_mock_pass": require_mock_pass,
        "required_adapters": required_adapters,
        "conformance_rows": int(len(normalized)),
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
    parser.add_argument(
        "--require-mock-pass",
        action="store_true",
        help="Require a `mock` adapter row with pass status to prevent false-green structural checks.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = verify_engine_readiness(
            conformance_matrix_path=Path(args.conformance_matrix),
            behavior_dir=Path(args.behavior_dir),
            allow_gpu_unavailable=bool(args.allow_gpu_unavailable),
            allow_nonpass_status=bool(args.allow_nonpass_status),
            require_mock_pass=bool(args.require_mock_pass),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"verify-engine-readiness failed: {exc}", file=sys.stderr)
        return 2
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
