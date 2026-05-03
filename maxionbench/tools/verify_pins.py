"""Verify MaxionBench scenario config pins."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.orchestration.config_schema import RunConfig, load_run_config

PINNED_BEIR_SUBSETS = ["scifact", "fiqa"]
PINNED_PORTABLE_EMBEDDINGS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}
PINNED_SCENARIOS = {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}


def verify_scenario_config_dir(config_dir: Path) -> dict[str, Any]:
    root = config_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Config directory does not exist: {root}")
    files = [path for path in sorted(root.glob("*.yaml")) if not path.name.startswith("._")]
    if not files:
        raise FileNotFoundError(f"No .yaml files found under: {root}")

    errors: list[dict[str, Any]] = []
    for cfg_path in files:
        cfg = load_run_config(cfg_path)
        errors.extend(_verify_common(cfg_path, cfg))
        errors.extend(_verify_scenario(cfg_path, cfg))

    return {
        "config_dir": str(root),
        "files_checked": len(files),
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _verify_common(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    _expect_equal(errors, path, "profile", cfg.profile, "maxionbench")
    if cfg.scenario == "s3_multi_hop":
        _expect_equal(errors, path, "dataset_bundle", cfg.dataset_bundle, "HOTPOT_PORTABLE")
    else:
        _expect_equal(errors, path, "dataset_bundle", cfg.dataset_bundle, "D4")
    _expect_equal(errors, path, "no_retry", cfg.no_retry, True)
    _expect_equal(errors, path, "rpc_baseline_requests", cfg.rpc_baseline_requests, 1000)
    _expect_equal(errors, path, "top_k", cfg.top_k, 10)
    _expect_equal(errors, path, "d4_beir_subsets", cfg.d4_beir_subsets, PINNED_BEIR_SUBSETS)
    if cfg.embedding_model is not None:
        expected_dim = PINNED_PORTABLE_EMBEDDINGS.get(str(cfg.embedding_model))
        if expected_dim is None:
            _error(errors, path, "embedding_model", sorted(PINNED_PORTABLE_EMBEDDINGS), cfg.embedding_model)
        else:
            _expect_equal(errors, path, "embedding_dim", cfg.embedding_dim, expected_dim)
            _expect_equal(errors, path, "vector_dim", cfg.vector_dim, expected_dim)
    return errors


def _verify_scenario(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if cfg.scenario not in PINNED_SCENARIOS:
        _error(errors, path, "scenario", sorted(PINNED_SCENARIOS), cfg.scenario)
        return errors
    if cfg.scenario == "s1_single_hop":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 1)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [1, 4, 8])
        _expect_equal(errors, path, "quality_target", cfg.quality_target, 0.25)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 50.0)
        _expect_equal(errors, path, "d4_max_docs", cfg.d4_max_docs, 50000)
    elif cfg.scenario == "s2_streaming_memory":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 8)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 2)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [8])
        _expect_equal(errors, path, "quality_target", cfg.quality_target, 0.25)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 120.0)
        _expect_equal(errors, path, "d4_max_docs", cfg.d4_max_docs, 50000)
    elif cfg.scenario == "s3_multi_hop":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 1)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [1, 4, 8])
        _expect_equal(errors, path, "quality_target", cfg.quality_target, 0.30)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 150.0)
    return errors


def _expect_equal(errors: list[dict[str, Any]], path: Path, field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        _error(errors, path, field, expected, actual)


def _error(errors: list[dict[str, Any]], path: Path, field: str, expected: Any, actual: Any) -> None:
    errors.append(
        {
            "file": str(path),
            "field": field,
            "expected": expected,
            "actual": actual,
            "message": f"{field} drift in {path.name}: expected {expected!r}, got {actual!r}",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify MaxionBench scenario config pins.")
    parser.add_argument("--config-dir", default="configs/scenarios_portable")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = verify_scenario_config_dir(Path(args.config_dir))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif not summary["pass"]:
        for error in summary["errors"]:
            print(error["message"])
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
