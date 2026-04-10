"""Local scratch-space preflight for workstation runs."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

from maxionbench.datasets.cache_integrity import resolve_expected_sha256_with_source, verify_file_sha256
from maxionbench.orchestration.config_schema import expand_env_placeholders

SAFETY_FACTOR_DEFAULT = 1.8
KNOWN_DATASET_BUNDLES = {"D1", "D2", "D3", "D4"}


def evaluate_local_preflight(
    *,
    config_path: Path,
    scratch_dir: Path,
    safety_factor: float = SAFETY_FACTOR_DEFAULT,
) -> dict[str, Any]:
    payload = _read_yaml_mapping(config_path)
    dataset_bundle = str(payload.get("dataset_bundle", "unknown"))
    manifest = _load_manifest(dataset_bundle)
    manifest_summary = _validate_manifest_requirements(dataset_bundle=dataset_bundle, manifest=manifest)
    integrity_summary = _verify_dataset_cache_integrity(
        config=payload,
        manifest=manifest,
        config_path=config_path,
    )

    estimates = _estimate_bytes(payload, manifest)
    required = int(safety_factor * (estimates["dataset_bytes"] + estimates["engine_bytes"] + estimates["temp_bytes"]))
    free = _free_bytes(scratch_dir)
    ok = free >= required and bool(integrity_summary["integrity_ok"]) and bool(manifest_summary["manifest_ok"])

    return {
        "ok": ok,
        "config_path": str(config_path),
        "scratch_dir": str(scratch_dir),
        "dataset_bundle": dataset_bundle,
        "free_bytes": int(free),
        "required_bytes": int(required),
        "dataset_bytes": int(estimates["dataset_bytes"]),
        "engine_bytes": int(estimates["engine_bytes"]),
        "temp_bytes": int(estimates["temp_bytes"]),
        "safety_factor": float(safety_factor),
        "fallback_config": estimates.get("fallback_config"),
        **manifest_summary,
        **integrity_summary,
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
    return dict(expand_env_placeholders(payload))


def _load_manifest(dataset_bundle: str) -> dict[str, Any] | None:
    root = Path(__file__).resolve().parents[1]
    candidate = root / "datasets" / "manifests" / f"{dataset_bundle.lower()}.yaml"
    if not candidate.exists():
        return None
    payload = _read_yaml_mapping(candidate)
    return payload


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _validate_manifest_requirements(dataset_bundle: str, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = str(dataset_bundle).upper()
    if bundle not in KNOWN_DATASET_BUNDLES:
        return {
            "manifest_ok": True,
            "manifest_error": None,
        }
    if manifest is None:
        return {
            "manifest_ok": False,
            "manifest_error": f"missing manifest for known dataset bundle {bundle}",
        }
    for key in ("approx_bytes_dataset", "approx_bytes_engine", "approx_bytes_temp"):
        value = manifest.get(key)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return {
                "manifest_ok": False,
                "manifest_error": f"manifest field {key} must be a positive integer for {bundle}",
            }
        if numeric <= 0:
            return {
                "manifest_ok": False,
                "manifest_error": f"manifest field {key} must be > 0 for {bundle}",
            }
    return {
        "manifest_ok": True,
        "manifest_error": None,
    }


def _verify_dataset_cache_integrity(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    config_path: Path,
) -> dict[str, Any]:
    mappings = [
        ("dataset_path", "dataset_path_sha256", "cache_sha256_dataset_path", "D1 dataset_path"),
        ("d2_base_fvecs_path", "d2_base_fvecs_sha256", "cache_sha256_d2_base_fvecs_path", "D2 d2_base_fvecs_path"),
        ("d2_query_fvecs_path", "d2_query_fvecs_sha256", "cache_sha256_d2_query_fvecs_path", "D2 d2_query_fvecs_path"),
        ("d2_gt_ivecs_path", "d2_gt_ivecs_sha256", "cache_sha256_d2_gt_ivecs_path", "D2 d2_gt_ivecs_path"),
        ("d4_crag_path", "d4_crag_sha256", "cache_sha256_d4_crag_path", "D4 d4_crag_path"),
    ]
    errors: list[str] = []
    checked: list[str] = []
    for path_key, cfg_checksum_key, manifest_checksum_key, label in mappings:
        raw_path = config.get(path_key)
        expected, source = resolve_expected_sha256_with_source(
            config_payload=config,
            manifest_payload=manifest,
            config_key=cfg_checksum_key,
            manifest_key=manifest_checksum_key,
            label=label,
        )
        if raw_path is None or raw_path == "":
            if expected is not None and isinstance(source, str) and source.startswith("config key "):
                errors.append(f"{label}: checksum provided but `{path_key}` is missing")
            continue
        if expected is None:
            continue
        candidate = _resolve_path(value=str(raw_path), config_path=config_path)
        try:
            verify_file_sha256(path=candidate, expected_sha256=expected, label=label)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        checked.append(label)
    return {
        "integrity_ok": len(errors) == 0,
        "integrity_error_count": len(errors),
        "integrity_checked_files": len(checked),
        "integrity_errors": errors,
    }


def _resolve_path(*, value: str, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="MaxionBench local storage preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--safety-factor", type=float, default=SAFETY_FACTOR_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = evaluate_local_preflight(
        config_path=Path(args.config).resolve(),
        scratch_dir=Path(args.scratch_dir).resolve(),
        safety_factor=float(args.safety_factor),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0 if bool(summary["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
