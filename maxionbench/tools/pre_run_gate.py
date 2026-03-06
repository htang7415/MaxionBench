"""Pre-run readiness gate for benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from maxionbench.orchestration.config_schema import load_run_config
from maxionbench.runtime.system_info import collect_system_info
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

    s5_runtime = _evaluate_s5_runtime_requirements(cfg)
    if s5_runtime is not None:
        summary["s5_reranker_runtime"] = s5_runtime
        if not bool(s5_runtime["pass"]):
            summary.update(
                {
                    "pass": False,
                    "skipped": False,
                    "reason": "s5 reranker runtime requirements not satisfied",
                }
            )
            return summary

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


def _evaluate_s5_runtime_requirements(cfg) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
    if str(getattr(cfg, "scenario", "")) != "s5_rerank":
        return None
    require_hf = bool(getattr(cfg, "s5_require_hf_backend", False))
    if not require_hf:
        return {
            "required": False,
            "pass": True,
            "errors": [],
        }

    env_enabled = _env_flag_true("MAXIONBENCH_ENABLE_HF_RERANKER")
    torch_installed = importlib.util.find_spec("torch") is not None
    transformers_installed = importlib.util.find_spec("transformers") is not None
    gpu_count = _detect_gpu_count()
    errors: list[str] = []
    if not env_enabled:
        errors.append("MAXIONBENCH_ENABLE_HF_RERANKER must be set to 1/true/on/yes for S5")
    if not torch_installed:
        errors.append("python package `torch` is required for S5 hf reranker backend")
    if not transformers_installed:
        errors.append("python package `transformers` is required for S5 hf reranker backend")
    if gpu_count < 1:
        errors.append("at least one NVIDIA GPU must be visible for S5 hf reranker backend")
    return {
        "required": True,
        "pass": len(errors) == 0,
        "env_enabled": env_enabled,
        "torch_installed": torch_installed,
        "transformers_installed": transformers_installed,
        "gpu_count": gpu_count,
        "errors": errors,
    }


def _env_flag_true(name: str) -> bool:
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _detect_gpu_count() -> int:
    try:
        payload = collect_system_info()
    except Exception:
        return 0
    try:
        return int(payload.get("gpu_count", 0) or 0)
    except Exception:
        return 0


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
