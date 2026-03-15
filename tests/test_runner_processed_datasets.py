from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION
from maxionbench.orchestration.runner import run_from_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for qid, did, rel in rows:
            handle.write(f"{qid}\t{did}\t{rel}\n")


def _make_processed_ann_dataset(root: Path, *, task_type: str = "ann") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": task_type,
            "metric": "angular",
        },
    )
    np.save(
        root / "base.npy",
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        ),
    )
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[0, 2], [1, 3]], dtype=np.int32))
    return root


def _make_processed_d4_root(root: Path) -> Path:
    beir = root / "beir" / "fiqa"
    beir.mkdir(parents=True, exist_ok=True)
    _write_json(
        beir / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        beir / "corpus.jsonl",
        [
            {"doc_id": "fiqa::doc::d1", "title": "Bond", "text": "bond market macro"},
            {"doc_id": "fiqa::doc::d2", "title": "Genome", "text": "genome medicine"},
        ],
    )
    _write_jsonl(
        beir / "queries.jsonl",
        [{"query_id": "fiqa::q::q1", "text": "bond market"}],
    )
    _write_qrels(beir / "qrels.tsv", [("fiqa::q::q1", "fiqa::doc::d1", 2)])
    return root


def test_runner_s1_uses_processed_d1_dataset(tmp_path: Path) -> None:
    processed = _make_processed_ann_dataset(tmp_path / "processed" / "D1" / "glove-100-angular")
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "processed-d1",
        "processed_dataset_path": str(processed),
        "seed": 9,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-d1"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 2,
        "num_vectors": 4,
        "num_queries": 2,
        "top_k": 2,
    }
    cfg_path = tmp_path / "cfg_d1.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    assert float(frame.iloc[0]["recall_at_10"]) >= 0.0


def test_runner_s1_uses_processed_d2_dataset(tmp_path: Path) -> None:
    processed = _make_processed_ann_dataset(tmp_path / "processed" / "D2" / "deep-image-96-angular")
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D2",
        "dataset_hash": "processed-d2",
        "processed_dataset_path": str(processed),
        "seed": 9,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-d2"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 2,
        "num_vectors": 4,
        "num_queries": 2,
        "top_k": 2,
    }
    cfg_path = tmp_path / "cfg_d2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    assert float(frame.iloc[0]["qps"]) >= 0.0


def test_runner_s4_uses_processed_d4_bundle(tmp_path: Path) -> None:
    processed_root = _make_processed_d4_root(tmp_path / "processed" / "D4")
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s4_hybrid",
        "dataset_bundle": "D4",
        "dataset_hash": "processed-d4",
        "processed_dataset_path": str(processed_root),
        "seed": 4,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-d4"),
        "quality_target": 0.35,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 150.0,
        "vector_dim": 8,
        "num_vectors": 10,
        "num_queries": 5,
        "top_k": 5,
        "rrf_k": 60,
        "s4_dense_candidates": 5,
        "s4_bm25_candidates": 5,
        "d4_beir_subsets": ["fiqa"],
        "d4_include_crag": False,
        "d4_max_docs": 10,
        "d4_max_queries": 5,
    }
    cfg_path = tmp_path / "cfg_d4.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(frame.iloc[0]["search_params_json"])
    assert payload["rag_ndcg_band"] in {"low", "medium", "high"}


def test_runner_rejects_processed_d3_for_s2_until_scenario_migration_finishes(tmp_path: Path) -> None:
    processed = _make_processed_ann_dataset(tmp_path / "processed" / "D3" / "yfcc-10M", task_type="filtered_ann")
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "processed-d3",
        "processed_dataset_path": str(processed),
        "seed": 7,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-d3"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 80.0,
        "vector_dim": 2,
        "num_vectors": 4,
        "num_queries": 2,
        "top_k": 2,
    }
    cfg_path = tmp_path / "cfg_d3.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="processed D3 datasets are not yet supported for S2/S3"):
        run_from_config(cfg_path, cli_overrides=None)
