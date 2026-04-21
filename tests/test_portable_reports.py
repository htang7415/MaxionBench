from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.runner import run_from_config
from maxionbench.scenarios import s2_streaming_memory as s2_streaming_memory_mod


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
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": "maxionbench-processed-v1",
                "task_type": "text_retrieval_strict",
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


def _portable_cfg(
    tmp_path: Path,
    *,
    name: str,
    scenario: str,
    processed_dataset_path: Path,
    dataset_bundle: str,
    budget_level: str,
) -> Path:
    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profile": "portable-agentic",
                "budget_level": budget_level,
                "engine": "mock",
                "engine_version": "0.1.0",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dim": 32,
                "c_llm_in": 0.15,
                "scenario": scenario,
                "dataset_bundle": dataset_bundle,
                "dataset_hash": f"{name}-fixture",
                "processed_dataset_path": str(processed_dataset_path),
                "seed": 42,
                "repeats": 1,
                "no_retry": True,
                "output_dir": str(tmp_path / "runs" / name),
                "quality_target": 0.0,
                "quality_targets": [0.0],
                "clients_read": 1,
                "clients_write": 0 if scenario != "s2_streaming_memory" else 1,
                "clients_grid": [1],
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


def test_portable_report_cli_exports_tables_and_figures(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    d4_root = tmp_path / "processed_d4"
    _write_processed_text_dataset(
        d4_root / "beir" / "scifact",
        docs=[
            {"doc_id": "scifact::doc::d1", "text": "ada discovered alpha particles"},
            {"doc_id": "scifact::doc::d2", "text": "bond markets matter"},
        ],
        queries=[{"query_id": "scifact::q::q1", "text": "alpha particles ada"}],
        qrels=[("scifact::q::q1", "scifact::doc::d1", 2)],
    )
    _write_processed_text_dataset(
        d4_root / "beir" / "fiqa",
        docs=[
            {"doc_id": "fiqa::doc::d1", "text": "bond market spreads widen"},
            {"doc_id": "fiqa::doc::d2", "text": "protein folding news"},
        ],
        queries=[{"query_id": "fiqa::q::q1", "text": "bond market spreads"}],
        qrels=[("fiqa::q::q1", "fiqa::doc::d1", 2)],
    )
    _write_processed_text_dataset(
        d4_root / "crag" / "small_slice",
        docs=[{"doc_id": "crag::doc::e1", "text": "fresh sports score update"}],
        queries=[{"query_id": "crag::q::q1", "text": "sports score update"}],
        qrels=[("crag::q::q1", "crag::doc::e1", 1)],
    )
    frames_root = tmp_path / "frames_portable"
    _write_processed_text_dataset(
        frames_root,
        docs=[
            {"doc_id": "frames::doc::d1", "text": "alpha was written by ada"},
            {"doc_id": "frames::doc::d2", "text": "alpha was published in journal b"},
        ],
        queries=[{"query_id": "frames::q::q1", "text": "who wrote alpha and where published"}],
        qrels=[
            ("frames::q::q1", "frames::doc::d1", 1),
            ("frames::q::q1", "frames::doc::d2", 1),
        ],
    )

    monkeypatch.setattr(s2_streaming_memory_mod, "_FRESHNESS_PROBE_DELAYS_S", (0.01, 0.05))
    monkeypatch.setattr(s2_streaming_memory_mod, "_VISIBILITY_POLL_INTERVAL_S", 0.01)

    run_from_config(_portable_cfg(tmp_path, name="s1_b0", scenario="s1_single_hop", processed_dataset_path=d4_root, dataset_bundle="D4", budget_level="b0"))
    run_from_config(_portable_cfg(tmp_path, name="s1_b1", scenario="s1_single_hop", processed_dataset_path=d4_root, dataset_bundle="D4", budget_level="b1"))
    run_from_config(_portable_cfg(tmp_path, name="s2_b1", scenario="s2_streaming_memory", processed_dataset_path=d4_root, dataset_bundle="D4", budget_level="b1"))
    run_from_config(_portable_cfg(tmp_path, name="s3_b2", scenario="s3_multi_hop", processed_dataset_path=frames_root, dataset_bundle="FRAMES_PORTABLE", budget_level="b2"))

    out_dir = tmp_path / "portable_report"
    code = cli_main(
        [
            "report",
            "--input",
            str(tmp_path / "runs"),
            "--mode",
            "portable-agentic",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0

    assert (out_dir / "portable_summary.csv").exists()
    assert (out_dir / "portable_winners.csv").exists()
    assert (out_dir / "portable_stability.csv").exists()
    assert (out_dir / "minimum_viable_deployment.csv").exists()
    assert (out_dir / "portable_summary.meta.json").exists()
    assert (out_dir / "portable_task_cost_by_budget.png").exists()
    assert (out_dir / "portable_task_cost_by_budget.meta.json").exists()
    assert (out_dir / "portable_budget_stability.png").exists()
    assert (out_dir / "portable_budget_stability.meta.json").exists()
    assert (out_dir / "portable_s2_freshness.png").exists()
    assert (out_dir / "portable_s2_freshness.meta.json").exists()

    winners = pd.read_csv(out_dir / "portable_winners.csv")
    deployment = pd.read_csv(out_dir / "minimum_viable_deployment.csv")
    stability = pd.read_csv(out_dir / "portable_stability.csv")
    task_cost_meta = json.loads((out_dir / "portable_task_cost_by_budget.meta.json").read_text(encoding="utf-8"))

    assert not winners.empty
    assert {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"} <= set(winners["scenario"].astype(str))
    assert not deployment.empty
    assert "workload_type" in deployment.columns
    assert task_cost_meta["mode"] == "portable-agentic"
    assert "rows_used" in task_cost_meta
    assert "spearman_rho" in stability.columns

