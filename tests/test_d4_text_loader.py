from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from maxionbench.datasets.loaders.d4_text import load_d4_from_local_bundles
from maxionbench.orchestration.runner import run_from_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _make_beir_subset(root: Path, name: str = "fiqa") -> Path:
    subset = root / name
    _write_jsonl(
        subset / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Finance", "text": "bond market macro"},
            {"_id": "d2", "title": "Science", "text": "genome medicine"},
            {"_id": "d3", "title": "Finance", "text": "equity risk"},
        ],
    )
    _write_jsonl(
        subset / "queries.jsonl",
        [
            {"_id": "q1", "text": "bond market"},
            {"_id": "q2", "text": "genome"},
        ],
    )
    qrels = subset / "qrels" / "test.tsv"
    qrels.parent.mkdir(parents=True, exist_ok=True)
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\n"
        "q1\td3\t1\n"
        "q2\td2\t2\n",
        encoding="utf-8",
    )
    return subset


def _make_crag_file(path: Path) -> Path:
    rows = [
        {
            "query_id": "1",
            "query": "weather report",
            "search_results": [
                {"doc_id": "u1", "title": "Forecast", "text": "weather report today", "relevance": 2},
                {"doc_id": "u2", "text": "finance update"},
            ],
        },
        {
            "query_id": "2",
            "query": "macro bond",
            "documents": [
                {"id": "u3", "text": "bond market news", "score": 1},
            ],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def test_load_d4_from_local_bundles_beir_only(tmp_path: Path) -> None:
    beir_root = tmp_path / "beir"
    _make_beir_subset(beir_root, "fiqa")

    ds = load_d4_from_local_bundles(
        vector_dim=16,
        seed=9,
        beir_root=beir_root,
        beir_subsets=["fiqa"],
        beir_split="test",
        include_crag=False,
        max_docs=10,
        max_queries=10,
    )
    assert ds.doc_vectors.shape[1] == 16
    assert len(ds.doc_ids) == 3
    assert len(ds.query_ids) == 2
    assert all(qid.startswith("fiqa::q::") for qid in ds.query_ids)
    assert all(doc_id.startswith("fiqa::doc::") for doc_id in ds.doc_ids)


def test_load_d4_from_local_bundles_beir_plus_crag(tmp_path: Path) -> None:
    beir_root = tmp_path / "beir"
    _make_beir_subset(beir_root, "fiqa")
    crag_path = _make_crag_file(tmp_path / "crag.jsonl.bz2")

    ds = load_d4_from_local_bundles(
        vector_dim=12,
        seed=3,
        beir_root=beir_root,
        beir_subsets=["fiqa"],
        crag_path=crag_path,
        include_crag=True,
        max_docs=6,
        max_queries=4,
    )
    assert ds.doc_vectors.shape == (len(ds.doc_ids), 12)
    assert ds.query_vectors.shape == (len(ds.query_ids), 12)
    assert any(doc_id.startswith("crag::doc::") for doc_id in ds.doc_ids)
    assert any(qid.startswith("crag::q::") for qid in ds.query_ids)
    for qid in ds.query_ids:
        assert qid in ds.qrels
        assert ds.qrels[qid]


def test_load_d4_from_local_bundles_enforces_crag_sha256(tmp_path: Path) -> None:
    beir_root = tmp_path / "beir"
    _make_beir_subset(beir_root, "fiqa")
    crag_path = _make_crag_file(tmp_path / "crag.jsonl.bz2")
    crag_sha = hashlib.sha256(crag_path.read_bytes()).hexdigest()

    ds = load_d4_from_local_bundles(
        vector_dim=8,
        seed=5,
        beir_root=beir_root,
        beir_subsets=["fiqa"],
        crag_path=crag_path,
        crag_expected_sha256=crag_sha,
        include_crag=True,
        max_docs=10,
        max_queries=10,
    )
    assert ds.doc_vectors.shape[1] == 8

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_d4_from_local_bundles(
            vector_dim=8,
            seed=5,
            beir_root=beir_root,
            beir_subsets=["fiqa"],
            crag_path=crag_path,
            crag_expected_sha256=("0" * 64),
            include_crag=True,
            max_docs=10,
            max_queries=10,
        )


def test_runner_s1_with_real_d4_bundle(tmp_path: Path) -> None:
    beir_root = tmp_path / "beir"
    _make_beir_subset(beir_root, "fiqa")

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_single_hop",
        "dataset_bundle": "D4",
        "dataset_hash": "local-beir",
        "seed": 4,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s4-real"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 2,
        "clients_write": 0,
        "clients_grid": [2],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 150.0,
        "vector_dim": 16,
        "num_vectors": 50,
        "num_queries": 10,
        "top_k": 10,
        "rrf_k": 60,
        "s4_dense_candidates": 20,
        "s4_bm25_candidates": 20,
        "d4_use_real_data": True,
        "d4_beir_root": str(beir_root),
        "d4_beir_subsets": ["fiqa"],
        "d4_beir_split": "test",
        "d4_include_crag": False,
        "d4_max_docs": 50,
        "d4_max_queries": 10,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(frame.iloc[0]["search_params_json"])
    assert payload["primary_quality_metric"] == "ndcg_at_10"


def test_runner_s1_enforces_crag_sha256_for_real_bundle(tmp_path: Path) -> None:
    beir_root = tmp_path / "beir"
    _make_beir_subset(beir_root, "fiqa")
    crag_path = _make_crag_file(tmp_path / "crag.jsonl.bz2")

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_single_hop",
        "dataset_bundle": "D4",
        "dataset_hash": "local-beir-crag",
        "seed": 4,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s4-real-crag"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 2,
        "clients_write": 0,
        "clients_grid": [2],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 150.0,
        "vector_dim": 16,
        "num_vectors": 50,
        "num_queries": 10,
        "top_k": 10,
        "rrf_k": 60,
        "s4_dense_candidates": 20,
        "s4_bm25_candidates": 20,
        "d4_use_real_data": True,
        "d4_beir_root": str(beir_root),
        "d4_beir_subsets": ["fiqa"],
        "d4_beir_split": "test",
        "d4_crag_path": str(crag_path),
        "d4_crag_sha256": "0" * 64,
        "d4_include_crag": True,
        "d4_max_docs": 50,
        "d4_max_queries": 10,
    }
    cfg_path = tmp_path / "cfg_crag_sha.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_from_config(cfg_path, cli_overrides=None)


def test_runner_s1_with_real_d4_bundle_relative_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    beir_root = data_root / "beir"
    _make_beir_subset(beir_root, "fiqa")

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_single_hop",
        "dataset_bundle": "D4",
        "dataset_hash": "local-beir-relative",
        "seed": 4,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s4-real-relative"),
        "quality_target": 0.0,
        "quality_targets": [0.0],
        "clients_read": 2,
        "clients_write": 0,
        "clients_grid": [2],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 150.0,
        "vector_dim": 16,
        "num_vectors": 50,
        "num_queries": 10,
        "top_k": 10,
        "rrf_k": 60,
        "s4_dense_candidates": 20,
        "s4_bm25_candidates": 20,
        "d4_use_real_data": True,
        "d4_beir_root": "data/beir",
        "d4_beir_subsets": ["fiqa"],
        "d4_beir_split": "test",
        "d4_include_crag": False,
        "d4_max_docs": 50,
        "d4_max_queries": 10,
    }
    cfg_path = tmp_path / "cfg_relative.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(frame.iloc[0]["search_params_json"])
    assert payload["primary_quality_metric"] == "ndcg_at_10"
