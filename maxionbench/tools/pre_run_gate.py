"""Pre-run readiness gate for benchmark execution."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from typing import Any

from maxionbench.conformance.provenance import conformance_provenance_path
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

    provenance = _evaluate_conformance_provenance(conformance_matrix_path.resolve())
    if provenance is not None:
        summary["conformance_provenance"] = provenance
        if not bool(provenance["pass"]):
            summary.update(
                {
                    "pass": False,
                    "skipped": False,
                    "reason": "conformance provenance validation failed",
                }
            )
            return summary

    readiness = verify_engine_readiness(
        conformance_matrix_path=conformance_matrix_path.resolve(),
        behavior_dir=behavior_dir.resolve(),
        allow_gpu_unavailable=allow_gpu_unavailable,
        target_adapter=str(cfg.engine),
    )
    summary["readiness"] = readiness
    summary["pass"] = bool(readiness["pass"])
    if not bool(readiness["pass"]):
        summary["reason"] = "engine readiness verification failed"
    return summary


def _evaluate_conformance_provenance(conformance_matrix_path: Path) -> dict[str, Any] | None:
    expected_runtime = str(os.environ.get("MAXIONBENCH_CONTAINER_RUNTIME", "")).strip().lower()
    expected_image = _normalized_path_str(os.environ.get("MAXIONBENCH_CONTAINER_IMAGE"))
    if not expected_runtime:
        return None

    provenance_path = conformance_provenance_path(conformance_matrix_path)
    summary: dict[str, Any] = {
        "required": True,
        "matrix_path": str(conformance_matrix_path),
        "provenance_path": str(provenance_path),
        "expected_container_runtime": expected_runtime,
        "expected_container_image": expected_image,
        "pass": False,
        "errors": [],
    }
    if not provenance_path.exists():
        summary["errors"].append(
            f"missing conformance provenance companion file `{provenance_path}` for `{conformance_matrix_path}`"
        )
        return summary

    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary["errors"].append(f"invalid conformance provenance JSON: {exc}")
        return summary
    if not isinstance(payload, dict):
        summary["errors"].append("conformance provenance payload must be a JSON object")
        return summary

    actual_runtime = str(payload.get("container_runtime", "")).strip().lower()
    actual_image = _normalized_path_str(payload.get("container_image"))
    summary["actual_container_runtime"] = actual_runtime
    summary["actual_container_image"] = actual_image
    summary["python_executable"] = str(payload.get("python_executable", "")).strip()
    summary["hostname"] = str(payload.get("hostname", "")).strip()
    summary["generated_at_utc"] = str(payload.get("generated_at_utc", "")).strip()

    if actual_runtime != expected_runtime:
        summary["errors"].append(
            f"conformance provenance runtime mismatch: expected `{expected_runtime}`, found `{actual_runtime or 'missing'}`"
        )
    if expected_image and actual_image != expected_image:
        summary["errors"].append(
            f"conformance provenance image mismatch: expected `{expected_image}`, found `{actual_image or 'missing'}`"
        )
    if not summary["errors"]:
        summary["pass"] = True
    return summary


def _normalized_path_str(raw: object) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return value


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
