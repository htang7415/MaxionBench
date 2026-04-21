from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from maxionbench.datasets.loaders.d4_synthetic import generate_d4_synthetic_dataset
from maxionbench.orchestration.runner import run_from_config
from maxionbench.scenarios.s4_hybrid import _ingest_dataset as ingest_s4_dataset
from maxionbench.scenarios.s5_rerank import _ingest_dataset as ingest_s5_dataset


class _BulkSpyAdapter:
    def __init__(self) -> None:
        self.bulk_sizes: list[int] = []
        self.flush_calls = 0

    def bulk_upsert(self, records) -> int:  # type: ignore[no-untyped-def]
        self.bulk_sizes.append(len(records))
        return len(records)

    def flush_or_commit(self) -> None:
        self.flush_calls += 1


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
        "s5_require_hf_backend": False,
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
        assert payload["reranker"]["backend"] in {"hf_cross_encoder", "heuristic_proxy"}
        assert payload["reranker"]["uses_qrels_supervision"] is False
    assert budgets.issubset({20, 40, 80})


def test_s4_and_s5_ingest_batch_large_d4_dataset() -> None:
    dataset = generate_d4_synthetic_dataset(num_docs=5_005, num_queries=4, vector_dim=12, seed=17)

    adapter_s4 = _BulkSpyAdapter()
    ingest_s4_dataset(adapter_s4, dataset)
    assert adapter_s4.bulk_sizes == [5_000, 5]
    assert adapter_s4.flush_calls == 1

    adapter_s5 = _BulkSpyAdapter()
    ingest_s5_dataset(adapter_s5, dataset)
    assert adapter_s5.bulk_sizes == [5_000, 5]
    assert adapter_s5.flush_calls == 1
