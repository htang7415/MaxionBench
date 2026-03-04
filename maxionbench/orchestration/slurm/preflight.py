"""Scratch-space preflight for Slurm benchmark jobs."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

SAFETY_FACTOR_DEFAULT = 1.8


def evaluate_preflight(
    *,
    config_path: Path,
    tmpdir: Path,
    safety_factor: float = SAFETY_FACTOR_DEFAULT,
) -> dict[str, Any]:
    payload = _read_yaml_mapping(config_path)
    dataset_bundle = str(payload.get("dataset_bundle", "unknown"))
    manifest = _load_manifest(dataset_bundle)

    estimates = _estimate_bytes(payload, manifest)
    required = int(safety_factor * (estimates["dataset_bytes"] + estimates["engine_bytes"] + estimates["temp_bytes"]))
    free = _free_bytes(tmpdir)
    ok = free >= required

    return {
        "ok": ok,
        "config_path": str(config_path),
        "tmpdir": str(tmpdir),
        "dataset_bundle": dataset_bundle,
        "free_bytes": int(free),
        "required_bytes": int(required),
        "dataset_bytes": int(estimates["dataset_bytes"]),
        "engine_bytes": int(estimates["engine_bytes"]),
        "temp_bytes": int(estimates["temp_bytes"]),
        "safety_factor": float(safety_factor),
        "fallback_config": estimates.get("fallback_config"),
    }


def _estimate_bytes(config: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest:
        dataset_bytes = int(manifest.get("approx_bytes_dataset", 0))
        engine_bytes = int(manifest.get("approx_bytes_engine", 0))
        temp_bytes = int(manifest.get("approx_bytes_temp", 0))
        if dataset_bytes > 0 and engine_bytes > 0 and temp_bytes > 0:
            return {
                "dataset_bytes": dataset_bytes,
                "engine_bytes": engine_bytes,
                "temp_bytes": temp_bytes,
                "fallback_config": manifest.get("fallback_config"),
            }

    num_vectors = int(config.get("num_vectors", 1))
    vector_dim = int(config.get("vector_dim", 1))
    base_vectors = max(1, num_vectors) * max(1, vector_dim) * 4
    # Conservative defaults when manifests are absent.
    dataset_bytes = int(base_vectors * 1.5)
    engine_bytes = int(config.get("estimated_engine_bytes", dataset_bytes * 1.2))
    temp_bytes = int(config.get("estimated_temp_bytes", dataset_bytes * 0.6))
    return {
        "dataset_bytes": dataset_bytes,
        "engine_bytes": engine_bytes,
        "temp_bytes": temp_bytes,
        "fallback_config": None,
    }


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return dict(payload)


def _load_manifest(dataset_bundle: str) -> dict[str, Any] | None:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "datasets" / "manifests" / f"{dataset_bundle.lower()}.yaml"
    if not candidate.exists():
        return None
    payload = _read_yaml_mapping(candidate)
    return payload


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def parse_args(argv: list[str] | None = None) -> ArgumentParser:
    parser = ArgumentParser(description="MaxionBench Slurm scratch preflight")
    parser.add_argument("--config", required=True, help="Scenario config path")
    parser.add_argument("--tmpdir", required=True, help="Scratch directory (typically $SLURM_TMPDIR)")
    parser.add_argument("--safety-factor", type=float, default=SAFETY_FACTOR_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parse_args(argv)
    args = parser.parse_args(argv)
    summary = evaluate_preflight(
        config_path=Path(args.config).resolve(),
        tmpdir=Path(args.tmpdir).resolve(),
        safety_factor=float(args.safety_factor),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if bool(summary["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
