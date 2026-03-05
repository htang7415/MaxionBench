"""Pre-run readiness gate for benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.orchestration.config_schema import load_run_config
from maxionbench.tools.verify_engine_readiness import verify_engine_readiness


def evaluate_pre_run_gate(
    *,
    config_path: Path,
    conformance_matrix_path: Path,
    behavior_dir: Path,
    allow_gpu_unavailable: bool = False,
    allow_mock: bool = True,
) -> dict[str, Any]:
    resolved_config = config_path.resolve()
    cfg = load_run_config(resolved_config)

    summary: dict[str, Any] = {
        "config_path": str(resolved_config),
        "engine": cfg.engine,
        "scenario": cfg.scenario,
        "dataset_bundle": cfg.dataset_bundle,
        "allow_gpu_unavailable": bool(allow_gpu_unavailable),
        "allow_mock": bool(allow_mock),
        "skipped": False,
    }

    if allow_mock and cfg.engine == "mock":
        summary.update(
            {
                "pass": True,
                "skipped": True,
                "reason": "mock engine is exempt from readiness gate",
            }
        )
        return summary

    readiness = verify_engine_readiness(
        conformance_matrix_path=conformance_matrix_path.resolve(),
        behavior_dir=behavior_dir.resolve(),
        allow_gpu_unavailable=allow_gpu_unavailable,
    )
    summary["readiness"] = readiness
    summary["pass"] = bool(readiness["pass"])
    if not bool(readiness["pass"]):
        summary["reason"] = "engine readiness verification failed"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run pre-run benchmark readiness gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--conformance-matrix", default="artifacts/conformance/conformance_matrix.csv")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    parser.add_argument("--allow-gpu-unavailable", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = evaluate_pre_run_gate(
        config_path=Path(args.config),
        conformance_matrix_path=Path(args.conformance_matrix),
        behavior_dir=Path(args.behavior_dir),
        allow_gpu_unavailable=bool(args.allow_gpu_unavailable),
        allow_mock=True,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print("pre-run gate passed")
        else:
            print("pre-run gate failed")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
