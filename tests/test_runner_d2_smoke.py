from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from maxionbench.orchestration.runner import run_from_config


def _write_fvecs(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    dim = arr.shape[1]
    packed = np.empty((arr.shape[0], dim + 1), dtype=np.int32)
    packed[:, 0] = dim
    packed[:, 1:] = arr.view(np.int32)
    packed.tofile(path)


def _write_ivecs(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int32)
    dim = arr.shape[1]
    packed = np.empty((arr.shape[0], dim + 1), dtype=np.int32)
    packed[:, 0] = dim
    packed[:, 1:] = arr
    packed.tofile(path)


def test_runner_s1_with_d2_paths(tmp_path: Path) -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0], [4.9, 5.1]], dtype=np.float32)
    gt = np.array([[0, 1], [2, 1]], dtype=np.int32)

    base_path = tmp_path / "base.fvecs"
    query_path = tmp_path / "query.fvecs"
    gt_path = tmp_path / "gt.ivecs"
    _write_fvecs(base_path, base)
    _write_fvecs(query_path, queries)
    _write_ivecs(gt_path, gt)

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D2",
        "dataset_hash": "synthetic-d2",
        "seed": 3,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 2,
        "num_vectors": 3,
        "num_queries": 2,
        "top_k": 2,
        "d2_base_fvecs_path": str(base_path),
        "d2_query_fvecs_path": str(query_path),
        "d2_gt_ivecs_path": str(gt_path),
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    assert frame.iloc[0]["dataset_bundle"] == "D2"
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["ground_truth_source"] == "bigann_ivecs"
    assert metadata["ground_truth_engine"] == "provided_ground_truth"
    assert int(metadata["ground_truth_k"]) == 2


def test_runner_rejects_invalid_checksum_format_in_config(tmp_path: Path) -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)
    base_path = tmp_path / "base.fvecs"
    query_path = tmp_path / "query.fvecs"
    _write_fvecs(base_path, base)
    _write_fvecs(query_path, queries)

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D2",
        "dataset_hash": "synthetic-d2",
        "seed": 3,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-bad-checksum"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 2,
        "num_vectors": 2,
        "num_queries": 1,
        "top_k": 1,
        "d2_base_fvecs_path": str(base_path),
        "d2_query_fvecs_path": str(query_path),
        "d2_base_fvecs_sha256": "not-a-sha",
    }
    cfg_path = tmp_path / "cfg_bad_checksum.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    with pytest.raises(ValueError, match="d2_base_fvecs_sha256"):
        run_from_config(cfg_path, cli_overrides=None)


def test_runner_records_dataset_cache_checksum_provenance_when_checksums_are_provided(tmp_path: Path) -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0], [4.9, 5.1]], dtype=np.float32)
    gt = np.array([[0, 1], [2, 1]], dtype=np.int32)

    base_path = tmp_path / "base_with_hash.fvecs"
    query_path = tmp_path / "query_with_hash.fvecs"
    gt_path = tmp_path / "gt_with_hash.ivecs"
    _write_fvecs(base_path, base)
    _write_fvecs(query_path, queries)
    _write_ivecs(gt_path, gt)

    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    query_sha = hashlib.sha256(query_path.read_bytes()).hexdigest()
    gt_sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D2",
        "dataset_hash": "synthetic-d2",
        "seed": 3,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-with-hash-provenance"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 2,
        "num_vectors": 3,
        "num_queries": 2,
        "top_k": 2,
        "d2_base_fvecs_path": str(base_path),
        "d2_query_fvecs_path": str(query_path),
        "d2_gt_ivecs_path": str(gt_path),
        "d2_base_fvecs_sha256": base_sha,
        "d2_query_fvecs_sha256": query_sha,
        "d2_gt_ivecs_sha256": gt_sha,
    }
    cfg_path = tmp_path / "cfg_with_hash_provenance.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    entries = metadata.get("dataset_cache_checksums")
    assert isinstance(entries, list)
    by_key = {str(item["path_key"]): item for item in entries}
    assert set(["d2_base_fvecs_path", "d2_query_fvecs_path", "d2_gt_ivecs_path"]).issubset(by_key.keys())
    for key, expected in [
        ("d2_base_fvecs_path", base_sha),
        ("d2_query_fvecs_path", query_sha),
        ("d2_gt_ivecs_path", gt_sha),
    ]:
        item = by_key[key]
        assert str(item["expected_sha256"]) == expected
        assert str(item["actual_sha256"]) == expected
        assert str(item["source"]).startswith("config key ")
