from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION, embedding_model_slug
from maxionbench.orchestration.config_schema import load_run_config
from maxionbench.orchestration.runner import run_from_config
from maxionbench.scenarios import s2_streaming_memory as s2_streaming_memory_mod
from maxionbench.tools.preprocess_hotpot_portable import preprocess_hotpot_portable


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_processed_text_dataset(
    path: Path,
    *,
    docs: list[dict],
    queries: list[dict],
    qrels: list[tuple[str, str, int]],
    task_type: str = "text_retrieval_strict",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": PROCESSED_SCHEMA_VERSION,
                "task_type": task_type,
                "name": path.name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(path / "corpus.jsonl", docs)
    _write_jsonl(path / "queries.jsonl", queries)
    with (path / "qrels.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in qrels:
            handle.write(f"{qid}\t{did}\t{score}\n")
    _write_precomputed_embeddings(path, docs=docs, queries=queries, qrels=qrels)


def _write_precomputed_embeddings(
    path: Path,
    *,
    docs: list[dict],
    queries: list[dict],
    qrels: list[tuple[str, str, int]],
    model_id: str = "BAAI/bge-small-en-v1.5",
    dim: int = 32,
) -> None:
    doc_ids = [str(row["doc_id"]) for row in docs]
    query_ids = [str(row["query_id"]) for row in queries]
    qrels_by_query: dict[str, list[str]] = {}
    for qid, did, score in qrels:
        if int(score) > 0:
            qrels_by_query.setdefault(str(qid), []).append(str(did))
    basis: dict[str, np.ndarray] = {}
    for idx, doc_id in enumerate(doc_ids):
        vec = np.zeros(dim, dtype=np.float32)
        vec[idx % dim] = 1.0
        basis[doc_id] = vec
    query_vectors: list[np.ndarray] = []
    for query_id in query_ids:
        vec = np.zeros(dim, dtype=np.float32)
        for doc_id in qrels_by_query.get(query_id, []):
            vec += basis[doc_id]
        if not np.any(vec):
            vec[0] = 1.0
        vec /= np.linalg.norm(vec) + 1e-12
        query_vectors.append(vec.astype(np.float32, copy=False))
    embedding_dir = path / "embeddings" / embedding_model_slug(model_id)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / "doc_vectors.npy", np.asarray([basis[doc_id] for doc_id in doc_ids], dtype=np.float32))
    np.save(embedding_dir / "query_vectors.npy", np.asarray(query_vectors, dtype=np.float32))
    doc_digest = json.dumps(doc_ids, separators=(",", ":")).encode("utf-8")
    query_digest = json.dumps(query_ids, separators=(",", ":")).encode("utf-8")
    (embedding_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": "maxionbench-text-embeddings-v1",
                "model_id": model_id,
                "dim": dim,
                "doc_count": len(doc_ids),
                "query_count": len(query_ids),
                "doc_ids_sha256": hashlib.sha256(doc_digest).hexdigest(),
                "query_ids_sha256": hashlib.sha256(query_digest).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _portable_cfg(
    tmp_path: Path,
    *,
    scenario: str,
    processed_dataset_path: Path,
    dataset_bundle: str,
    budget_level: str = "b0",
    clients_grid: list[int] | None = None,
    clients_read: int = 1,
    clients_write: int = 0,
) -> Path:
    cfg_path = tmp_path / f"{scenario}.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profile": "maxionbench",
                "budget_level": budget_level,
                "engine": "mock",
                "engine_version": "0.1.0",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dim": 32,
                "c_llm_in": 0.15,
                "scenario": scenario,
                "dataset_bundle": dataset_bundle,
                "dataset_hash": f"{scenario}-fixture",
                "processed_dataset_path": str(processed_dataset_path),
                "seed": 42,
                "repeats": 9,
                "no_retry": True,
                "output_dir": str(tmp_path / "run" / scenario),
                "quality_target": 0.0,
                "quality_targets": [0.0],
                "clients_read": clients_read,
                "clients_write": clients_write,
                "clients_grid": list(clients_grid or [clients_read]),
                "search_sweep": [{"hnsw_ef": 16}, {"hnsw_ef": 32}],
                "rpc_baseline_requests": 5,
                "sla_threshold_ms": 50.0 if scenario == "s1_single_hop" else 120.0 if scenario == "s2_streaming_memory" else 150.0,
                "vector_dim": 32,
                "num_vectors": 10,
                "num_queries": 10,
                "top_k": 10,
                "d4_beir_subsets": ["scifact", "fiqa"],
                "d4_include_crag": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_portable_configs_load_with_new_schema_fields() -> None:
    for rel_path, scenario in [
        ("configs/scenarios_portable/s1_single_hop.yaml", "s1_single_hop"),
        ("configs/scenarios_portable/s2_streaming_memory.yaml", "s2_streaming_memory"),
        ("configs/scenarios_portable/s3_multi_hop.yaml", "s3_multi_hop"),
    ]:
        cfg = load_run_config(Path(rel_path))
        assert cfg.profile == "maxionbench"
        assert cfg.budget_level == "b1"
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding_dim == 384
        assert cfg.c_llm_in == 0.15
        assert cfg.scenario == scenario
    s1_cfg = load_run_config(Path("configs/scenarios_portable/s1_single_hop.yaml"))
    s2_cfg = load_run_config(Path("configs/scenarios_portable/s2_streaming_memory.yaml"))
    s3_cfg = load_run_config(Path("configs/scenarios_portable/s3_multi_hop.yaml"))
    assert s1_cfg.clients_grid == [1, 4, 8]
    assert s1_cfg.d4_max_docs == 50000
    assert (s2_cfg.clients_read, s2_cfg.clients_write, s2_cfg.clients_grid) == (8, 2, [8])
    assert s2_cfg.d4_max_docs == 50000
    assert s3_cfg.clients_grid == [1, 4, 8]
    assert s1_cfg.rpc_baseline_requests == 1000
    assert s2_cfg.rpc_baseline_requests == 1000
    assert s3_cfg.rpc_baseline_requests == 1000


def test_run_from_config_portable_s1_smoke(tmp_path: Path) -> None:
    root = tmp_path / "processed_d4"
    _write_processed_text_dataset(
        root / "beir" / "scifact",
        docs=[
            {"doc_id": "scifact::doc::d1", "text": "ada discovered alpha particles"},
            {"doc_id": "scifact::doc::d2", "text": "bond markets matter"},
        ],
        queries=[{"query_id": "scifact::q::q1", "text": "alpha particles ada"}],
        qrels=[("scifact::q::q1", "scifact::doc::d1", 2)],
    )
    _write_processed_text_dataset(
        root / "beir" / "fiqa",
        docs=[
            {"doc_id": "fiqa::doc::d1", "text": "bond market spreads widen"},
            {"doc_id": "fiqa::doc::d2", "text": "protein folding news"},
        ],
        queries=[{"query_id": "fiqa::q::q1", "text": "bond market spreads"}],
        qrels=[("fiqa::q::q1", "fiqa::doc::d1", 2)],
    )
    cfg_path = _portable_cfg(tmp_path, scenario="s1_single_hop", processed_dataset_path=root, dataset_bundle="D4", clients_grid=[1, 4, 8])

    out_dir = run_from_config(cfg_path)

    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(str(frame.iloc[0]["search_params_json"]))
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((out_dir / "config_resolved.yaml").read_text(encoding="utf-8"))

    assert payload["primary_quality_metric"] == "ndcg_at_10"
    assert payload["task_cost_est"] >= 0.0
    assert metadata["ground_truth_metric"] == "ndcg_at_10"
    assert metadata["budget_level"] == "b0"
    assert int(metadata["repeats"]) == 1
    assert int(resolved["warmup_s"]) == 10
    assert int(resolved["steady_state_s"]) == 10
    assert sorted(set(frame["clients_read"].astype(int).tolist())) == [1, 4, 8]
    assert str(frame.iloc[0]["budget_level"]) == "b0"
    assert str(frame.iloc[0]["embedding_model"]) == "BAAI/bge-small-en-v1.5"
    assert float(frame.iloc[0]["task_cost_est"]) >= 0.0


def test_run_from_config_portable_s2_smoke(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "processed_d4"
    _write_processed_text_dataset(
        root / "beir" / "scifact",
        docs=[
            {"doc_id": "scifact::doc::d1", "text": "ada discovered alpha particles"},
            {"doc_id": "scifact::doc::d2", "text": "bond markets matter"},
        ],
        queries=[{"query_id": "scifact::q::q1", "text": "alpha particles ada"}],
        qrels=[("scifact::q::q1", "scifact::doc::d1", 2)],
    )
    _write_processed_text_dataset(
        root / "beir" / "fiqa",
        docs=[
            {"doc_id": "fiqa::doc::d1", "text": "bond market spreads widen"},
            {"doc_id": "fiqa::doc::d2", "text": "protein folding news"},
        ],
        queries=[{"query_id": "fiqa::q::q1", "text": "bond market spreads"}],
        qrels=[("fiqa::q::q1", "fiqa::doc::d1", 2)],
    )
    _write_processed_text_dataset(
        root / "crag" / "small_slice",
        docs=[
            {"doc_id": "crag::doc::e1", "text": "fresh sports score update"},
            {"doc_id": "crag::doc::e2", "text": "fresh market close update"},
        ],
        queries=[
            {"query_id": "crag::q::q1", "text": "sports score update"},
            {"query_id": "crag::q::q2", "text": "market close update"},
        ],
        qrels=[
            ("crag::q::q1", "crag::doc::e1", 1),
            ("crag::q::q2", "crag::doc::e2", 1),
        ],
        task_type="text_retrieval_weak",
    )
    monkeypatch.setattr(s2_streaming_memory_mod, "_FRESHNESS_PROBE_DELAYS_S", (0.01, 0.05))
    monkeypatch.setattr(s2_streaming_memory_mod, "_VISIBILITY_POLL_INTERVAL_S", 0.01)
    cfg_path = _portable_cfg(
        tmp_path,
        scenario="s2_streaming_memory",
        processed_dataset_path=root,
        dataset_bundle="D4",
        clients_read=2,
        clients_write=1,
    )

    out_dir = run_from_config(cfg_path)

    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(str(frame.iloc[0]["search_params_json"]))
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert payload["primary_quality_metric"] == "ndcg_at_10"
    assert payload["freshness_hit_at_1s"] >= 0.0
    assert payload["freshness_hit_at_5s"] >= 0.0
    assert payload["stale_answer_rate_at_5s"] >= 0.0
    assert payload["p95_visibility_latency_ms"] >= 0.0
    assert metadata["ground_truth_metric"] == "ndcg_at_10"
    assert float(frame.iloc[0]["freshness_hit_at_1s"]) >= 0.0
    assert float(frame.iloc[0]["freshness_hit_at_5s"]) >= 0.0
    assert float(frame.iloc[0]["stale_answer_rate_at_5s"]) >= 0.0
    assert float(frame.iloc[0]["p95_visibility_latency_ms"]) >= 0.0


def test_run_from_config_portable_s3_smoke(tmp_path: Path) -> None:
    root = tmp_path / "hotpot_portable"
    _write_processed_text_dataset(
        root,
        docs=[
            {"doc_id": "hotpot::doc::d1", "text": "alpha was written by ada"},
            {"doc_id": "hotpot::doc::d2", "text": "alpha was published in journal b"},
            {"doc_id": "hotpot::doc::d3", "text": "beta is a protein"},
        ],
        queries=[
            {"query_id": "hotpot::q::q1", "text": "who wrote alpha and where published"},
            {"query_id": "hotpot::q::q2", "text": "what is beta"},
        ],
        qrels=[
            ("hotpot::q::q1", "hotpot::doc::d1", 1),
            ("hotpot::q::q1", "hotpot::doc::d2", 1),
            ("hotpot::q::q2", "hotpot::doc::d3", 1),
        ],
    )
    cfg_path = _portable_cfg(tmp_path, scenario="s3_multi_hop", processed_dataset_path=root, dataset_bundle="HOTPOT_PORTABLE", clients_read=2)

    out_dir = run_from_config(cfg_path)

    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) >= 1
    payload = json.loads(str(frame.iloc[0]["search_params_json"]))
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert payload["primary_quality_metric"] == "evidence_coverage@10"
    assert payload["evidence_coverage_at_5"] >= 0.0
    assert payload["evidence_coverage_at_10"] >= 0.0
    assert payload["evidence_coverage_at_20"] >= 0.0
    assert payload["task_cost_est"] >= 0.0
    assert metadata["ground_truth_metric"] == "evidence_coverage@10"
    assert float(frame.iloc[0]["evidence_coverage_at_10"]) >= 0.0


def test_preprocess_hotpot_portable_builds_processed_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "hotpot_dev_distractor_v1.json"
    out_dir = tmp_path / "processed" / "hotpot_portable"
    input_path.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "Who wrote Alpha and where was it published?",
                    "answer": "Ada",
                    "supporting_facts": [["PageA", 0], ["PageB", 0]],
                    "context": [
                        ["PageA", ["Alpha was written by Ada.", "Extra sentence."]],
                        ["PageB", ["Alpha was published in Journal B."]],
                        ["PageC", ["Distractor text."]],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = preprocess_hotpot_portable(input_path=input_path, out_dir=out_dir)

    assert summary["dataset_name"] == "hotpotqa-maxionbench"
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "corpus.jsonl").exists()
    assert (out_dir / "queries.jsonl").exists()
    assert (out_dir / "qrels.tsv").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "checksums.json").exists()
