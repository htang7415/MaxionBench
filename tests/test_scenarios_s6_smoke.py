from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from maxionbench.orchestration.runner import run_from_config


def test_s6_fusion_smoke(tmp_path: Path) -> None:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s6_fusion",
        "dataset_bundle": "D4",
        "dataset_hash": "synthetic-d4",
        "seed": 44,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s6-run"),
        "quality_target": 0.35,
        "clients_read": 4,
        "clients_write": 0,
        "clients_grid": [4],
        "search_sweep": [{"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 180.0,
        "vector_dim": 24,
        "num_vectors": 700,
        "num_queries": 30,
        "top_k": 10,
        "rrf_k": 60,
        "s6_dense_a_candidates": 120,
        "s6_dense_b_candidates": 120,
        "s6_bm25_candidates": 120,
    }
    cfg_path = tmp_path / "s6.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1

    modes = set()
    for payload_json in frame["search_params_json"].tolist():
        payload = json.loads(payload_json)
        assert payload["rag_ndcg_band"] in {"low", "medium", "high"}
        modes.add(payload["mode"])
    assert modes.issubset({"s6a_dense_dense_rrf", "s6b_dense_bm25_rrf"})
