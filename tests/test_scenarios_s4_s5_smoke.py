from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from maxionbench.orchestration.runner import run_from_config


def _run_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / f"{cfg['scenario']}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)
    return run_from_config(path, cli_overrides=None)


def test_s4_hybrid_smoke(tmp_path: Path) -> None:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s4_hybrid",
        "dataset_bundle": "D4",
        "dataset_hash": "synthetic-d4",
        "seed": 13,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s4-run"),
        "quality_target": 0.35,
        "clients_read": 4,
        "clients_write": 0,
        "clients_grid": [4],
        "search_sweep": [{"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 150.0,
        "vector_dim": 24,
        "num_vectors": 600,
        "num_queries": 30,
        "top_k": 10,
        "rrf_k": 60,
        "s4_dense_candidates": 120,
        "s4_bm25_candidates": 120,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1

    modes = set()
    for row in frame["search_params_json"].tolist():
        payload = json.loads(row)
        assert payload["rag_ndcg_band"] in {"low", "medium", "high"}
        modes.add(payload["mode"])
    assert modes.issubset({"dense_only", "bm25_dense_rrf"})


def test_s5_rerank_smoke(tmp_path: Path) -> None:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s5_rerank",
        "dataset_bundle": "D4",
        "dataset_hash": "synthetic-d4",
        "seed": 21,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s5-run"),
        "quality_target": 0.35,
        "clients_read": 4,
        "clients_write": 0,
        "clients_grid": [4],
        "search_sweep": [{"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 300.0,
        "vector_dim": 24,
        "num_vectors": 700,
        "num_queries": 35,
        "top_k": 10,
        "s5_candidate_budgets": [20, 40, 80],
        "s5_reranker_model_id": "BAAI/bge-reranker-base",
        "s5_reranker_revision_tag": "2026-03-04",
        "s5_reranker_max_seq_len": 512,
        "s5_reranker_precision": "fp16",
        "s5_reranker_batch_size": 32,
        "s5_reranker_truncation": "right",
    }
    out_dir = _run_cfg(tmp_path, cfg)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    budgets = set()
    for payload_json in frame["search_params_json"].tolist():
        payload = json.loads(payload_json)
        assert payload["rag_ndcg_band"] in {"low", "medium", "high"}
        budgets.add(int(payload["candidate_budget"]))
        assert payload["reranker"]["revision_tag"] == "2026-03-04"
    assert budgets.issubset({20, 40, 80})
