from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
