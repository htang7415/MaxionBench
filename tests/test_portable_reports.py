from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.datasets.loaders.processed import embedding_model_slug
from maxionbench.orchestration.runner import run_from_config
from maxionbench.reports.portable_exports import (
    _cost_formula_table,
    _cost_sensitivity_table,
    _decision_surface_table,
    _decision_error_ablation_table,
    _engine_configuration_table,
    _extract_portable_frame,
    _index_search_configuration,
    _latency_distribution_table,
    _neurips_main_results_table,
    _minimum_viable_deployment_sensitivity_table,
    _minimum_viable_deployment_table,
    _portable_decision_table,
    _quality_floor_survivor_table,
    _s3_all_evidence_hit_table,
    _s2_post_insert_examples_table,
    _s2_write_diagnostic_table,
    _spearman_rank_correlation,
    _strict_decision_margin_table,
    _winner_rows,
    generate_portable_report_bundle,
)
from maxionbench.scenarios import s2_streaming_memory as s2_streaming_memory_mod
from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS


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
                "profile": "maxionbench",
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
    hotpot_root = tmp_path / "hotpot_portable"
    _write_processed_text_dataset(
        hotpot_root,
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
    run_from_config(_portable_cfg(tmp_path, name="s3_b2", scenario="s3_multi_hop", processed_dataset_path=hotpot_root, dataset_bundle="HOTPOT_PORTABLE", budget_level="b2"))

    conformance_dir = tmp_path / "artifacts" / "conformance"
    conformance_dir.mkdir(parents=True)
    pd.DataFrame([{"adapter": adapter, "status": "pass"} for adapter in ("mock", *REQUIRED_ADAPTERS)]).to_csv(
        conformance_dir / "conformance_matrix.csv",
        index=False,
    )
    behavior_dir = tmp_path / "docs" / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    out_dir = tmp_path / "portable_report"
    code = cli_main(
        [
            "report",
            "--input",
            str(tmp_path / "runs"),
            "--mode",
            "maxionbench",
            "--out",
            str(out_dir),
            "--conformance-matrix",
            str(conformance_dir / "conformance_matrix.csv"),
            "--behavior-dir",
            str(behavior_dir),
        ]
    )
    assert code == 0

    assert (out_dir / "portable_summary.csv").exists()
    assert (out_dir / "portable_winners.csv").exists()
    assert (out_dir / "portable_stability.csv").exists()
    assert (out_dir / "minimum_viable_deployment.csv").exists()
    assert (out_dir / "minimum_viable_deployment_sensitivity.csv").exists()
    assert (out_dir / "portable_decision_table.csv").exists()
    assert (out_dir / "portable_decision_table.tex").exists()
    assert (out_dir / "decision_surface.csv").exists()
    assert (out_dir / "decision_surface.tex").exists()
    assert (out_dir / "s2_write_diagnostics.csv").exists()
    assert (out_dir / "s2_write_diagnostics.tex").exists()
    assert (out_dir / "neurips_main_results.csv").exists()
    assert (out_dir / "neurips_main_results.tex").exists()
    assert (out_dir / "decision_error_ablation.csv").exists()
    assert (out_dir / "decision_error_ablation.tex").exists()
    assert (out_dir / "quality_floor_survivors.csv").exists()
    assert (out_dir / "quality_floor_survivors.tex").exists()
    assert (out_dir / "cost_formula.csv").exists()
    assert (out_dir / "cost_formula.tex").exists()
    assert (out_dir / "cost_sensitivity.csv").exists()
    assert (out_dir / "cost_sensitivity.tex").exists()
    assert (out_dir / "latency_distribution.csv").exists()
    assert (out_dir / "latency_distribution.tex").exists()
    assert (out_dir / "strict_decision_margins.csv").exists()
    assert (out_dir / "strict_decision_margins.tex").exists()
    assert (out_dir / "engine_configuration.csv").exists()
    assert (out_dir / "engine_configuration.tex").exists()
    assert (out_dir / "s3_all_evidence_hit.csv").exists()
    assert (out_dir / "s3_all_evidence_hit.tex").exists()
    assert (out_dir / "portable_support_table.csv").exists()
    assert (out_dir / "portable_support_table.tex").exists()
    assert (out_dir / "portable_summary.meta.json").exists()
    assert (out_dir / "maxionbench_decision_audit_conceptual.svg").exists()
    assert (out_dir / "maxionbench_decision_audit_conceptual.meta.json").exists()
    assert (out_dir / "portable_decision_surface.svg").exists()
    assert (out_dir / "portable_decision_surface.meta.json").exists()
    assert (out_dir / "s3_paired_audit_forest.svg").exists()
    assert (out_dir / "s3_paired_audit_forest.meta.json").exists()
    assert (out_dir / "portable_task_cost_by_budget.svg").exists()
    assert (out_dir / "portable_task_cost_by_budget.meta.json").exists()
    assert (out_dir / "portable_budget_stability.svg").exists()
    assert (out_dir / "portable_budget_stability.meta.json").exists()
    assert (out_dir / "portable_s2_post_insert_retrievability.svg").exists()
    assert (out_dir / "portable_s2_post_insert_retrievability.meta.json").exists()
    assert (out_dir / "portable_mvd_sensitivity.svg").exists()
    assert (out_dir / "portable_mvd_sensitivity.meta.json").exists()
    for figure_name in (
        "maxionbench_decision_audit_conceptual",
        "portable_decision_surface",
        "s3_paired_audit_forest",
        "portable_task_cost_by_budget",
        "portable_budget_stability",
        "portable_s2_post_insert_retrievability",
        "portable_mvd_sensitivity",
    ):
        assert (out_dir / f"{figure_name}.svg").exists()
        assert (out_dir / f"{figure_name}.pdf").exists()
        assert (out_dir / f"{figure_name}.png").exists()

    summary = pd.read_csv(out_dir / "portable_summary.csv")
    winners = pd.read_csv(out_dir / "portable_winners.csv")
    deployment = pd.read_csv(out_dir / "minimum_viable_deployment.csv")
    decision = pd.read_csv(out_dir / "portable_decision_table.csv")
    decision_surface = pd.read_csv(out_dir / "decision_surface.csv")
    s2_write_diagnostics = pd.read_csv(out_dir / "s2_write_diagnostics.csv")
    s2_post_insert_examples = pd.read_csv(out_dir / "s2_post_insert_examples.csv")
    stability = pd.read_csv(out_dir / "portable_stability.csv")
    latency = pd.read_csv(out_dir / "latency_distribution.csv")
    strict_margins = pd.read_csv(out_dir / "strict_decision_margins.csv")
    engine_configuration = pd.read_csv(out_dir / "engine_configuration.csv")
    support = pd.read_csv(out_dir / "portable_support_table.csv")
    task_cost_meta = json.loads((out_dir / "portable_task_cost_by_budget.meta.json").read_text(encoding="utf-8"))
    latency_tex = (out_dir / "latency_distribution.tex").read_text(encoding="utf-8")
    engine_configuration_tex = (out_dir / "engine_configuration.tex").read_text(encoding="utf-8")

    assert not winners.empty
    assert "post_insert_hit_at_10_5s" in summary.columns
    assert "freshness_hit_at_5s" not in summary.columns
    assert "post_insert_hit_at_10_5s" in winners.columns
    assert "freshness_hit_at_5s" not in winners.columns
    assert {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"} <= set(winners["scenario"].astype(str))
    assert not deployment.empty
    assert "workload_type" in deployment.columns
    assert not decision.empty
    assert "strict_p99_engine" in decision.columns
    assert "quality_winner_engine" in decision.columns
    assert {"workload", "role", "strict_p99_pass", "p99_max_ms"} <= set(decision_surface.columns)
    assert "B2 decision-surface rows" in (out_dir / "decision_surface.tex").read_text(encoding="utf-8")
    assert {"post_insert_hit_at_10_1s", "p95_visibility_latency_ms", "event_count"} <= set(s2_write_diagnostics.columns)
    assert "not a direct insert-index visibility-latency measurement" in (out_dir / "s2_write_diagnostics.tex").read_text(encoding="utf-8")
    assert {"category", "event_count", "event_rate", "example_query_id", "interpretation"} <= set(s2_post_insert_examples.columns)
    assert "post-insert example outcomes" in (out_dir / "s2_post_insert_examples.tex").read_text(encoding="utf-8")
    assert {"clients_read_write", "boundary"} <= set(latency.columns)
    assert {
        "workload",
        "strict_choice",
        "next_strict_candidate",
        "delta_quality",
        "delta_task_cost_est",
        "delta_p99_ms",
    } <= set(strict_margins.columns)
    assert "R/W clients" in latency_tex
    assert "Boundary" in latency_tex
    assert "adapter.query plus top-k materialization" in latency_tex
    assert {"index_search_configuration", "distance_metric", "flush_commit_path"} <= set(engine_configuration.columns)
    assert "Index/search" in engine_configuration_tex
    assert "Metric" in engine_configuration_tex
    assert "Flush/commit" in engine_configuration_tex
    faiss_config = engine_configuration.loc[engine_configuration["engine"].astype(str) == "faiss-cpu"].iloc[0]
    assert "exact IndexFlatIP" in str(faiss_config["index_search_configuration"])
    assert "ignored by flat index" in _index_search_configuration(
        engine="faiss-cpu",
        config={"metric": "ip", "index_params": {}, "search_sweep": [{"hnsw_ef": 32}], "top_k": 10},
    )
    assert not support.empty
    assert set(REQUIRED_ADAPTERS) <= set(support["engine"].astype(str))
    assert "reportable" in support.columns
    assert "included_in_report" in support.columns
    assert "\\label{tab:portable-support}" in (out_dir / "portable_support_table.tex").read_text(encoding="utf-8")
    assert task_cost_meta["mode"] == "maxionbench"
    assert "rows_used" in task_cost_meta
    assert "spearman_rho" in stability.columns


def test_neurips_main_results_table_includes_quality_and_post_insert_ci_fields() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 8,
                "rank_within_budget": 1,
                "engine": "engine-s2",
                "embedding_model": "emb-s2",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "freshness_hit_at_1s": 0.84,
                "freshness_hit_at_5s": 0.84,
                "event_count": 500,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s2_streaming_memory",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 8,
                "rank_within_budget": 1,
                "engine": "engine-s2",
                "embedding_model": "emb-s2",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.52,
                "freshness_hit_at_1s": 0.84,
                "freshness_hit_at_5s": 0.84,
                "event_count": 500,
                "p99_ms": 12.0,
                "qps": 90.0,
                "task_cost_est": 1.2,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_pair": "b0->b2",
                "spearman_rho": 0.4,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )

    table = _neurips_main_results_table(frame=winners, winners=winners, stability=stability)
    row = table.iloc[0]

    assert row["primary_quality_mean"] == pytest.approx(0.51)
    assert row["primary_quality_ci95_low"] <= row["primary_quality_mean"] <= row["primary_quality_ci95_high"]
    assert row["post_insert_hit_at_10_5s_mean"] == pytest.approx(0.84)
    assert row["post_insert_hit_at_10_5s_ci95_low"] < row["post_insert_hit_at_10_5s_ci95_high"]
    assert row["post_insert_event_count"] == 500
    assert row["decision_stability_note"] == "top-1 stable despite full-rank noise"


def test_s2_post_insert_examples_table_counts_archived_outcomes(tmp_path: Path) -> None:
    observation_path = tmp_path / "s2_freshness.jsonl"
    _write_jsonl(
        observation_path,
        [
            {
                "observation_type": "freshness",
                "event_index": 1,
                "query_id": "crag_small_slice::q::hit",
                "target_doc_id": "crag_small_slice::doc::hit_p0_c0",
                "freshness_hit_at_1s": 1,
                "freshness_hit_at_5s": 1,
                "visibility_latency_ms": 0.1,
            },
            {
                "observation_type": "freshness",
                "event_index": 2,
                "query_id": "crag_small_slice::q::late",
                "target_doc_id": "crag_small_slice::doc::late_p0_c0",
                "freshness_hit_at_1s": 0,
                "freshness_hit_at_5s": 1,
                "visibility_latency_ms": 1000.0,
            },
            {
                "observation_type": "freshness",
                "event_index": 3,
                "query_id": "crag_small_slice::q::miss",
                "target_doc_id": "crag_small_slice::doc::miss_p0_c0",
                "freshness_hit_at_1s": 0,
                "freshness_hit_at_5s": 0,
                "visibility_latency_ms": 5000.0,
            },
        ],
    )
    frame = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 8,
                "clients_write": 2,
                "rank_within_budget": 1,
                "engine": "engine-s2",
                "embedding_model": "emb-s2",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "freshness_hit_at_1s": 2 / 3,
                "freshness_hit_at_5s": 2 / 3,
                "event_count": 3,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 1.0,
                "observation_path": str(observation_path),
                "__run_path": str(tmp_path),
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_pair": "b0->b2",
                "spearman_rho": 1.0,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )
    decision = _portable_decision_table(winners=frame, stability=stability)

    table = _s2_post_insert_examples_table(frame=frame, decision=decision)
    counts = dict(zip(table["category"], table["event_count"], strict=True))

    assert counts["retrieved by 1s"] == 1
    assert counts["recovered between 1s and 5s"] == 1
    assert counts["missed by 5s"] == 1
    assert table["event_rate"].sum() == pytest.approx(1.0)
    missed = table.loc[table["category"] == "missed by 5s"].iloc[0]
    assert missed["example_query_id"] == "crag_small_slice::q::miss"
    assert "censored at 5s" in str(missed["interpretation"])


def test_neurips_main_results_p99_matches_latency_distribution_strict_choice() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 1,
                "clients_write": 0,
                "rank_within_budget": 1,
                "engine": "engine-s1",
                "embedding_model": "emb-s1",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "p50_ms": 4.0,
                "p95_ms": 6.0,
                "p99_ms": 10.0,
                "qps": 100.0,
                "measure_requests": 10,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 4,
                "clients_write": 0,
                "rank_within_budget": 1,
                "engine": "engine-s1",
                "embedding_model": "emb-s1",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.52,
                "p50_ms": 5.0,
                "p95_ms": 7.0,
                "p99_ms": 12.0,
                "qps": 90.0,
                "measure_requests": 10,
                "task_cost_est": 1.1,
            },
        ]
    )
    broader_frame = pd.concat(
        [
            winners,
            pd.DataFrame(
                [
                    {
                        **winners.iloc[0].to_dict(),
                        "clients_read": 8,
                        "p99_ms": 99.0,
                        "task_cost_est": 1.2,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_pair": "b0->b2",
                "spearman_rho": 1.0,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )

    main = _neurips_main_results_table(frame=broader_frame, winners=winners, stability=stability)
    decision = _portable_decision_table(winners=winners, stability=stability)
    latency = _latency_distribution_table(winners=winners, decision=decision)
    strict_latency = latency.loc[latency["row_role"] == "strict choice"].iloc[0]

    assert main.iloc[0]["p99_ms_max"] == pytest.approx(strict_latency["p99_max_ms"])
    assert main.iloc[0]["p99_ms_max"] == pytest.approx(12.0)


def test_strict_decision_margin_p99_delta_uses_latency_distribution() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 1,
                "clients_write": 0,
                "rank_within_budget": 1,
                "engine": "faiss-cpu",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "p50_ms": 4.0,
                "p95_ms": 6.0,
                "p99_ms": 10.0,
                "qps": 100.0,
                "measure_requests": 10,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 4,
                "clients_write": 0,
                "rank_within_budget": 1,
                "engine": "faiss-cpu",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "p50_ms": 5.0,
                "p95_ms": 7.0,
                "p99_ms": 12.0,
                "qps": 90.0,
                "measure_requests": 10,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 1,
                "clients_write": 0,
                "rank_within_budget": 2,
                "engine": "lancedb-inproc",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "p50_ms": 14.0,
                "p95_ms": 26.0,
                "p99_ms": 30.0,
                "qps": 80.0,
                "measure_requests": 10,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 4,
                "clients_write": 0,
                "rank_within_budget": 2,
                "engine": "lancedb-inproc",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "p50_ms": 15.0,
                "p95_ms": 36.0,
                "p99_ms": 40.0,
                "qps": 70.0,
                "measure_requests": 10,
                "task_cost_est": 1.0,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_pair": "b0->b2",
                "spearman_rho": 1.0,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )

    decision = _portable_decision_table(winners=winners, stability=stability)
    latency = _latency_distribution_table(winners=winners, decision=decision)
    margins = _strict_decision_margin_table(winners=winners, latency_distribution=latency)
    row = margins.iloc[0]

    assert row["delta_p99_ms"] == pytest.approx(28.0)
    assert row["delta_p99_ms"] == pytest.approx(row["candidate_p99_max_ms"] - row["strict_p99_max_ms"])
    assert row["interpretation"] == "cost/quality tie; p99 tie-break"


def test_neurips_main_results_table_prefers_archived_observations(tmp_path: Path) -> None:
    observation_path = tmp_path / "observations.jsonl"
    _write_jsonl(
        observation_path,
        [
            {"observation_type": "quality", "query_id": "q1", "ndcg_at_10": 0.25},
            {"observation_type": "quality", "query_id": "q2", "ndcg_at_10": 0.75},
            {"observation_type": "freshness", "query_id": "e1", "freshness_hit_at_1s": 1, "freshness_hit_at_5s": 1},
            {"observation_type": "freshness", "query_id": "e2", "freshness_hit_at_1s": 0, "freshness_hit_at_5s": 1},
        ],
    )
    winners = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 8,
                "rank_within_budget": 1,
                "engine": "engine-s2",
                "embedding_model": "emb-s2",
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.50,
                "freshness_hit_at_1s": 0.0,
                "freshness_hit_at_5s": 0.0,
                "event_count": 500,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 1.0,
                "observation_path": str(observation_path),
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_pair": "b0->b2",
                "spearman_rho": 1.0,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )

    table = _neurips_main_results_table(frame=winners, winners=winners, stability=stability)
    row = table.iloc[0]

    assert row["primary_quality_mean"] == pytest.approx(0.5)
    assert row["primary_quality_samples"] == 2
    assert str(row["primary_quality_ci_method"]).startswith("query-level bootstrap")
    assert row["post_insert_hit_at_10_1s_mean"] == pytest.approx(0.5)
    assert row["post_insert_hit_at_10_5s_mean"] == pytest.approx(1.0)
    assert row["post_insert_event_count"] == 2
    assert str(row["post_insert_ci_method"]).startswith("Wilson binomial CI from archived per-event")


def test_winner_rows_keeps_clients_read_dimension() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_level": "b1",
                "budget_sort": 1,
                "clients_read": 1,
                "engine": "engine-a",
                "embedding_model": "emb",
                "task_cost_est": 3.0,
                "p99_ms": 10.0,
                "qps": 100.0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b1",
                "budget_sort": 1,
                "clients_read": 8,
                "engine": "engine-a",
                "embedding_model": "emb",
                "task_cost_est": 1.0,
                "p99_ms": 12.0,
                "qps": 90.0,
            },
        ]
    )

    winners = _winner_rows(frame=frame)

    assert sorted(winners["clients_read"].astype(int).tolist()) == [1, 8]


def test_winner_rows_aggregates_latency_across_repeat_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 1,
                "engine": "engine-a",
                "embedding_model": "emb",
                "task_cost_est": 1.0,
                "p50_ms": 2.0,
                "p95_ms": 4.0,
                "p99_ms": 10.0,
                "qps": 100.0,
                "measure_requests": 20,
                "repeat_idx": 0,
            },
            {
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "clients_read": 1,
                "engine": "engine-a",
                "embedding_model": "emb",
                "task_cost_est": 1.0,
                "p50_ms": 4.0,
                "p95_ms": 8.0,
                "p99_ms": 250.0,
                "qps": 50.0,
                "measure_requests": 30,
                "repeat_idx": 1,
            },
        ]
    )

    winners = _winner_rows(frame=frame)
    row = winners.iloc[0]

    assert row["p50_ms"] == pytest.approx(3.0)
    assert row["p95_ms"] == pytest.approx(6.0)
    assert row["p99_min_ms"] == pytest.approx(10.0)
    assert row["p99_ms"] == pytest.approx(250.0)
    assert row["qps"] == pytest.approx(75.0)
    assert int(row["measure_requests"]) == 50
    assert int(row["latency_row_count"]) == 2
    assert int(row["latency_repeat_count"]) == 2


def test_spearman_rank_correlation_is_nan_for_single_observation() -> None:
    assert np.isnan(_spearman_rank_correlation([1.0], [1.0]))


def test_extract_portable_frame_falls_back_when_string_columns_are_none() -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "scenario": "s1_single_hop",
                "engine": "mock",
                "repeat_idx": 0,
                "quality_target": 0.0,
                "search_params_json": json.dumps(
                    {
                        "budget_level": "b1",
                        "embedding_model": "BAAI/bge-small-en-v1.5",
                        "task_cost_est": 1.25,
                    },
                    sort_keys=True,
                ),
                "budget_level": None,
                "embedding_model": None,
                "__meta_profile": "maxionbench",
                "__meta_budget_level": "b0",
                "__meta_embedding_model": "fallback-embedding",
            }
        ]
    )

    portable = _extract_portable_frame(frame=frame)

    assert portable.iloc[0]["budget_level"] == "b1"
    assert portable.iloc[0]["embedding_model"] == "BAAI/bge-small-en-v1.5"


def test_minimum_viable_deployment_table_includes_freshness_for_s2_rows() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s2_streaming_memory",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 1,
                "engine": "engine-s2",
                "embedding_model": "emb-s2",
                "primary_quality_metric": "answer_f1",
                "primary_quality_value": 0.812,
                "freshness_hit_at_5s": 0.975,
                "p99_ms": 84.5,
                "qps": 12.0,
                "task_cost_est": 0.456789,
            }
        ]
    )

    deployment = _minimum_viable_deployment_table(winners=winners)

    assert deployment.iloc[0]["why"] == "answer_f1=0.812, post_insert_hit@10,5s=0.975, p99_mean=84.5ms, p99_max=84.5ms, task_cost=0.456789"


def test_minimum_viable_deployment_sensitivity_exposes_latency_threshold_effect() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 1,
                "engine": "engine-cheap-slow",
                "embedding_model": "emb",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.82,
                "p99_ms": 300.0,
                "qps": 10.0,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 2,
                "engine": "engine-fast",
                "embedding_model": "emb",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.81,
                "p99_ms": 80.0,
                "qps": 20.0,
                "task_cost_est": 2.0,
            },
        ]
    )

    sensitivity = _minimum_viable_deployment_sensitivity_table(winners=winners)
    by_threshold = {
        str(row["p99_max_threshold_ms"]): row["minimum_engine"]
        for row in sensitivity.to_dict(orient="records")
    }

    assert by_threshold["100.0"] == "engine-fast"
    assert by_threshold["200.0"] == "engine-fast"
    assert by_threshold["500.0"] == "engine-cheap-slow"
    assert by_threshold["none"] == "engine-cheap-slow"


def test_portable_decision_table_separates_latency_cost_and_quality_choices() -> None:
    winners = pd.DataFrame(
        [
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 1,
                "engine": "engine-cheap-slow",
                "embedding_model": "emb-small",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.82,
                "p99_ms": 300.0,
                "qps": 10.0,
                "task_cost_est": 1.0,
            },
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 2,
                "engine": "engine-fast",
                "embedding_model": "emb-small",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.81,
                "p99_ms": 80.0,
                "qps": 20.0,
                "task_cost_est": 2.0,
            },
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 3,
                "engine": "engine-quality",
                "embedding_model": "emb-base",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.90,
                "p99_ms": 90.0,
                "qps": 15.0,
                "task_cost_est": 3.0,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s3_multi_hop",
                "budget_pair": "b0->b2",
                "spearman_rho": 0.3,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )

    table = _portable_decision_table(winners=winners, stability=stability)
    row = table.iloc[0]

    assert row["strict_p99_engine"] == "engine-fast"
    assert row["unconstrained_cost_engine"] == "engine-cheap-slow"
    assert row["quality_winner_engine"] == "engine-quality"
    assert row["decision_stability_note"] == "top-1 stable despite full-rank noise"


def test_task3_decision_audit_tables_expose_required_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "candidate"
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "engine": "faiss-cpu",
                "engine_version": "cpu",
                "metric": "ip",
                "search_sweep": [{"hnsw_ef": 32}],
                "top_k": 10,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"engine": "faiss-cpu", "engine_version": "cpu"}) + "\n",
        encoding="utf-8",
    )
    observation_path = run_dir / "logs" / "observations" / "s3.jsonl"
    _write_jsonl(
        observation_path,
        [
            {"observation_type": "quality", "evidence_coverage_at_10": 1.0},
            {"observation_type": "quality", "evidence_coverage_at_10": 0.5},
        ],
    )
    winners = pd.DataFrame(
        [
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 1,
                "clients_read": 1,
                "clients_write": 0,
                "engine": "faiss-cpu",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.82,
                "evidence_coverage_at_10": 0.82,
                "p50_ms": 5.0,
                "p95_ms": 8.0,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 10.0,
                "retrieval_cost_est": 1.0,
                "embedding_cost_est": 2.0,
                "avg_retrieved_input_tokens": 40.0,
                "c_llm_in": 0.15,
                "measure_requests": 50,
                "embedding_dim": 384,
                "observation_path": str(observation_path),
                "__run_path": str(run_dir),
            },
            {
                "scenario": "s3_multi_hop",
                "budget_level": "b2",
                "budget_sort": 2,
                "rank_within_budget": 2,
                "clients_read": 1,
                "clients_write": 0,
                "engine": "qdrant",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "primary_quality_metric": "evidence_coverage@10",
                "primary_quality_value": 0.80,
                "evidence_coverage_at_10": 0.80,
                "p50_ms": 4.0,
                "p95_ms": 7.0,
                "p99_ms": 9.0,
                "qps": 90.0,
                "task_cost_est": 11.0,
                "retrieval_cost_est": 1.2,
                "embedding_cost_est": 2.0,
                "avg_retrieved_input_tokens": 42.0,
                "c_llm_in": 0.15,
                "measure_requests": 50,
                "embedding_dim": 384,
                "observation_path": "",
                "__run_path": "",
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "scenario": "s3_multi_hop",
                "budget_pair": "b0->b2",
                "spearman_rho": 0.2,
                "top1_agreement": 1.0,
                "top2_agreement": 1.0,
            }
        ]
    )
    decision = _portable_decision_table(winners=winners, stability=stability)
    decision_surface = _decision_surface_table(winners=winners, decision=decision)
    s2_write = _s2_write_diagnostic_table(winners=winners, decision=decision)

    decision_error = _decision_error_ablation_table(decision=decision, stability=stability)
    quality_floor = _quality_floor_survivor_table(winners=winners)
    cost_formula = _cost_formula_table()
    cost_sensitivity = _cost_sensitivity_table(winners=winners)
    latency = _latency_distribution_table(winners=winners, decision=decision)
    strict_margins = _strict_decision_margin_table(winners=winners, latency_distribution=latency)
    support = pd.DataFrame(
        [
            {
                "engine": "faiss-cpu",
                "behavior_card": "faiss_cpu.md",
                "included_in_report": True,
            }
        ]
    )
    engine_config = _engine_configuration_table(frame=winners, support=support)
    all_evidence = _s3_all_evidence_hit_table(winners=winners, decision=decision)

    assert {
        "workload",
        "role",
        "engine",
        "embedding_model",
        "quality_metric",
        "quality_value",
        "post_insert_hit_at_10_5s",
        "task_cost_est",
        "p99_max_ms",
        "strict_p99_pass",
        "source_path",
    } <= set(decision_surface.columns)
    assert {
        "role",
        "engine",
        "embedding_model",
        "ndcg_at_10",
        "post_insert_hit_at_10_1s",
        "post_insert_hit_at_10_5s",
        "p95_visibility_latency_ms",
        "event_count",
        "errors",
        "p99_max_ms",
        "source_path",
    } <= set(s2_write.columns)
    assert {
        "missing_protocol_component",
        "wrong_conclusion_caused_by_omission",
        "manuscript_evidence",
        "source_path",
    } <= set(decision_error.columns)
    assert {
        "workload",
        "quality_metric",
        "quality_floor_source",
        "floor_value",
        "strict_p99_survivor_count_b2",
        "source_path",
    } <= set(quality_floor.columns)
    assert {"term", "meaning", "unit", "value_source", "source_path"} <= set(cost_formula.columns)
    assert {
        "workload",
        "candidate_role",
        "c_llm_in_multiplier",
        "sensitivity_task_cost_est",
        "selection_changes_from_main",
        "source_path",
    } <= set(cost_sensitivity.columns)
    assert {
        "workload",
        "engine",
        "embedding_model",
        "clients_read_write",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "p99_min_ms",
        "p99_max_ms",
        "latency_row_count",
        "latency_repeat_count",
        "latency_observations",
        "boundary",
        "source_path",
    } <= set(latency.columns)
    assert {
        "workload",
        "strict_choice",
        "strict_engine",
        "strict_embedding_model",
        "next_strict_candidate",
        "candidate_engine",
        "candidate_embedding_model",
        "quality_metric",
        "delta_quality",
        "delta_task_cost_est",
        "strict_p99_max_ms",
        "candidate_p99_max_ms",
        "delta_p99_ms",
        "interpretation",
        "source_path",
    } <= set(strict_margins.columns)
    assert {
        "engine",
        "mode",
        "version",
        "index_search_configuration",
        "distance_metric",
        "embedding_dimension",
        "process_model",
        "flush_commit_path",
        "included_in_reported_matrix",
        "source_path",
    } <= set(engine_config.columns)
    assert {
        "row_role",
        "engine",
        "embedding_model",
        "evidence_coverage_at_10",
        "all_evidence_hit_at_10",
        "query_level_observations",
        "method",
        "source_path",
    } <= set(all_evidence.columns)
    assert all_evidence.iloc[0]["all_evidence_hit_at_10"] == pytest.approx(0.5)


def test_generate_portable_report_bundle_requires_conformance_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "qdrant"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "engine": "qdrant",
                "embedding_model": "emb",
                "quality_target": 0.0,
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.5,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 1.0,
                "freshness_hit_at_5s": float("nan"),
                "stale_answer_rate_at_5s": float("nan"),
                "evidence_coverage_at_10": float("nan"),
                "clients_read": 1,
                "repeat_idx": 0,
                "search_params_json": "{}",
                "__meta_profile": "maxionbench",
            }
        ]
    ).to_parquet(run_dir / "results.parquet", index=False)
    (run_dir / "run_status.json").write_text(json.dumps({"status": "success"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="require --conformance-matrix"):
        generate_portable_report_bundle(
            input_dir=tmp_path / "runs",
            out_dir=tmp_path / "out",
            conformance_matrix_path=None,
            behavior_dir=tmp_path / "behavior",
        )


def test_support_table_filters_non_reportable_engines_from_report(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    qdrant_run = runs_root / "qdrant"
    pgvector_run = runs_root / "pgvector"
    qdrant_run.mkdir(parents=True)
    pgvector_run.mkdir(parents=True)

    frame = pd.DataFrame(
        [
            {
                "run_id": "qdrant-run",
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "engine": "qdrant",
                "embedding_model": "emb",
                "quality_target": 0.0,
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.5,
                "p99_ms": 10.0,
                "qps": 100.0,
                "task_cost_est": 1.0,
                "freshness_hit_at_5s": float("nan"),
                "stale_answer_rate_at_5s": float("nan"),
                "evidence_coverage_at_10": float("nan"),
                "clients_read": 1,
                "repeat_idx": 0,
                "search_params_json": "{}",
                "__meta_profile": "maxionbench",
            },
            {
                "run_id": "pgvector-run",
                "scenario": "s1_single_hop",
                "budget_level": "b2",
                "engine": "pgvector",
                "embedding_model": "emb",
                "quality_target": 0.0,
                "primary_quality_metric": "ndcg_at_10",
                "primary_quality_value": 0.49,
                "p99_ms": 12.0,
                "qps": 90.0,
                "task_cost_est": 1.1,
                "freshness_hit_at_5s": float("nan"),
                "stale_answer_rate_at_5s": float("nan"),
                "evidence_coverage_at_10": float("nan"),
                "clients_read": 1,
                "repeat_idx": 0,
                "search_params_json": "{}",
                "__meta_profile": "maxionbench",
            },
        ]
    )
    frame.iloc[[0]].to_parquet(qdrant_run / "results.parquet", index=False)
    frame.iloc[[1]].to_parquet(pgvector_run / "results.parquet", index=False)
    (qdrant_run / "run_status.json").write_text(json.dumps({"status": "success"}) + "\n", encoding="utf-8")
    (pgvector_run / "run_status.json").write_text(json.dumps({"status": "success"}) + "\n", encoding="utf-8")

    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame(
        [
            {"adapter": "qdrant", "status": "pass"},
            {"adapter": "pgvector", "status": "fail"},
            {"adapter": "faiss-cpu", "status": "pass"},
            {"adapter": "lancedb-inproc", "status": "fail"},
            {"adapter": "lancedb-service", "status": "fail"},
        ]
    ).to_csv(matrix_path, index=False)
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    out_dir = tmp_path / "out"
    generate_portable_report_bundle(
        input_dir=runs_root,
        out_dir=out_dir,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )

    winners = pd.read_csv(out_dir / "portable_winners.csv")
    support = pd.read_csv(out_dir / "portable_support_table.csv")

    assert winners["engine"].tolist() == ["qdrant"]
    pgvector_row = support.loc[support["engine"] == "pgvector"].iloc[0]
    assert bool(pgvector_row["reportable"]) is False
    assert bool(pgvector_row["included_in_report"]) is False
