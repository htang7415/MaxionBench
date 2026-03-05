"""Verify pinned scenario config values from project/prompt specs."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

from maxionbench.orchestration.config_schema import RunConfig, load_run_config

PINNED_BEIR_SUBSETS = ["trec-covid", "nfcorpus", "fiqa", "scifact", "hotpotqa"]
PINNED_RAG_CRAG_SOURCE = "facebookresearch/CRAG"
PINNED_RAG_CRAG_PATH = "data/crag_task_1_and_2_dev_v4.jsonl.bz2"
PINNED_ANN_QUALITY_TARGETS = [0.80, 0.90, 0.95]
D3_50M_MIN_VECTORS = 50_000_000


def verify_scenario_config_dir(config_dir: Path) -> dict[str, Any]:
    root = config_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Config directory does not exist: {root}")

    files = sorted(root.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No .yaml files found under: {root}")

    errors: list[dict[str, Any]] = []
    for cfg_path in files:
        cfg = load_run_config(cfg_path)
        errors.extend(_verify_common_pins(cfg_path, cfg))
        errors.extend(_verify_scenario_pins(cfg_path, cfg))
        errors.extend(_verify_d3_pins(cfg_path, cfg))
        errors.extend(_verify_d4_real_data_pins(cfg_path, cfg))

    return {
        "config_dir": str(root),
        "files_checked": len(files),
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _verify_common_pins(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    _expect_equal(errors, path, "no_retry", cfg.no_retry, True)
    _expect_equal(errors, path, "warmup_s", cfg.warmup_s, 120)
    _expect_equal(errors, path, "steady_state_s", cfg.steady_state_s, 300)
    _expect_equal(errors, path, "rpc_baseline_requests", cfg.rpc_baseline_requests, 1000)
    _expect_equal(errors, path, "top_k", cfg.top_k, 10)
    _expect_equal(errors, path, "c_ref_vcpu", cfg.c_ref_vcpu, 96.0)
    _expect_equal(errors, path, "g_ref_gpu", cfg.g_ref_gpu, 1.0)
    _expect_equal(errors, path, "r_ref_gib", cfg.r_ref_gib, 512.0)
    _expect_equal(errors, path, "d_ref_tb", cfg.d_ref_tb, 7.68)
    _expect_equal(errors, path, "w_c", cfg.w_c, 0.25)
    _expect_equal(errors, path, "w_g", cfg.w_g, 0.25)
    _expect_equal(errors, path, "w_r", cfg.w_r, 0.25)
    _expect_equal(errors, path, "w_d", cfg.w_d, 0.25)
    return errors


def _verify_scenario_pins(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    scenario = cfg.scenario
    if scenario == "calibrate_d3":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 1)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [1])
        return errors

    if scenario == "s1_ann_frontier":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 1)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [1, 8, 32, 64])
        _expect_equal(errors, path, "quality_targets", _normalized_float_list(cfg.quality_targets), PINNED_ANN_QUALITY_TARGETS)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 50.0)
        return errors

    if scenario == "s2_filtered_ann":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 32)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [32])
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 80.0)
        return errors

    if scenario == "s3_churn_smooth":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 32)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 8)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [32])
        _expect_equal(errors, path, "lambda_req_s", cfg.lambda_req_s, 1000.0)
        _expect_equal(errors, path, "s3_read_rate", cfg.s3_read_rate, 800.0)
        _expect_equal(errors, path, "s3_insert_rate", cfg.s3_insert_rate, 100.0)
        _expect_equal(errors, path, "s3_update_rate", cfg.s3_update_rate, 50.0)
        _expect_equal(errors, path, "s3_delete_rate", cfg.s3_delete_rate, 50.0)
        _expect_equal(errors, path, "maintenance_interval_s", cfg.maintenance_interval_s, 60.0)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 120.0)
        return errors

    if scenario == "s3b_churn_bursty":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 32)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 8)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [32])
        _expect_equal(errors, path, "lambda_req_s", cfg.lambda_req_s, 1000.0)
        _expect_equal(errors, path, "s3_read_rate", cfg.s3_read_rate, 800.0)
        _expect_equal(errors, path, "s3_insert_rate", cfg.s3_insert_rate, 100.0)
        _expect_equal(errors, path, "s3_update_rate", cfg.s3_update_rate, 50.0)
        _expect_equal(errors, path, "s3_delete_rate", cfg.s3_delete_rate, 50.0)
        _expect_equal(errors, path, "maintenance_interval_s", cfg.maintenance_interval_s, 60.0)
        _expect_equal(errors, path, "s3b_on_s", cfg.s3b_on_s, 30.0)
        _expect_equal(errors, path, "s3b_off_s", cfg.s3b_off_s, 90.0)
        _expect_equal(errors, path, "s3b_on_write_mult", cfg.s3b_on_write_mult, 8.0)
        _expect_equal(errors, path, "s3b_off_write_mult", cfg.s3b_off_write_mult, 0.25)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 120.0)
        return errors

    if scenario == "s4_hybrid":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 16)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [16])
        _expect_equal(errors, path, "rrf_k", cfg.rrf_k, 60)
        _expect_equal(errors, path, "s4_dense_candidates", cfg.s4_dense_candidates, 200)
        _expect_equal(errors, path, "s4_bm25_candidates", cfg.s4_bm25_candidates, 200)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 150.0)
        return errors

    if scenario == "s5_rerank":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 16)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [16])
        _expect_equal(errors, path, "s5_candidate_budgets", cfg.s5_candidate_budgets, [50, 200, 1000])
        _expect_equal(errors, path, "s5_reranker_model_id", cfg.s5_reranker_model_id, "BAAI/bge-reranker-base")
        _expect_equal(errors, path, "s5_reranker_revision_tag", cfg.s5_reranker_revision_tag, "2026-03-04")
        _expect_equal(errors, path, "s5_reranker_max_seq_len", cfg.s5_reranker_max_seq_len, 512)
        _expect_equal(errors, path, "s5_reranker_precision", cfg.s5_reranker_precision, "fp16")
        _expect_equal(errors, path, "s5_reranker_batch_size", cfg.s5_reranker_batch_size, 32)
        _expect_equal(errors, path, "s5_reranker_truncation", cfg.s5_reranker_truncation, "right")
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 300.0)
        return errors

    if scenario == "s6_fusion":
        _expect_equal(errors, path, "clients_read", cfg.clients_read, 16)
        _expect_equal(errors, path, "clients_write", cfg.clients_write, 0)
        _expect_equal(errors, path, "clients_grid", cfg.clients_grid, [16])
        _expect_equal(errors, path, "rrf_k", cfg.rrf_k, 60)
        _expect_equal(errors, path, "s6_dense_a_candidates", cfg.s6_dense_a_candidates, 200)
        _expect_equal(errors, path, "s6_dense_b_candidates", cfg.s6_dense_b_candidates, 200)
        _expect_equal(errors, path, "s6_bm25_candidates", cfg.s6_bm25_candidates, 200)
        _expect_equal(errors, path, "sla_threshold_ms", cfg.sla_threshold_ms, 180.0)
        return errors

    errors.append(
        {
            "file": str(path),
            "field": "scenario",
            "expected": "known scenario",
            "actual": scenario,
            "message": f"unsupported scenario for pin verification: {scenario}",
        }
    )
    return errors


def _verify_d3_pins(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    if cfg.dataset_bundle.upper() != "D3":
        return []
    errors: list[dict[str, Any]] = []
    expected_k = 8192 if int(cfg.num_vectors) >= D3_50M_MIN_VECTORS else 4096
    _expect_equal(errors, path, "d3_k_clusters", cfg.d3_k_clusters, expected_k)
    _expect_equal(errors, path, "d3_num_tenants", cfg.d3_num_tenants, 100)
    _expect_equal(errors, path, "d3_num_acl_buckets", cfg.d3_num_acl_buckets, 16)
    _expect_equal(errors, path, "d3_num_time_buckets", cfg.d3_num_time_buckets, 52)
    _expect_equal(errors, path, "d3_beta_tenant", cfg.d3_beta_tenant, 0.75)
    _expect_equal(errors, path, "d3_beta_acl", cfg.d3_beta_acl, 0.70)
    _expect_equal(errors, path, "d3_beta_time", cfg.d3_beta_time, 0.65)
    return errors


def _verify_d4_real_data_pins(path: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    if cfg.dataset_bundle.upper() != "D4" or not cfg.d4_use_real_data:
        return []
    errors: list[dict[str, Any]] = []
    _expect_equal(errors, path, "d4_beir_subsets", cfg.d4_beir_subsets, PINNED_BEIR_SUBSETS)
    _expect_equal(errors, path, "d4_beir_split", cfg.d4_beir_split, "test")
    _expect_equal(errors, path, "d4_max_docs", cfg.d4_max_docs, 200000)
    _expect_equal(errors, path, "d4_max_queries", cfg.d4_max_queries, 5000)
    _expect_equal(errors, path, "d4_include_crag", cfg.d4_include_crag, True)
    if cfg.d4_include_crag:
        _expect_equal(errors, path, "d4_crag_source", cfg.d4_crag_source, PINNED_RAG_CRAG_SOURCE)
        _expect_equal(errors, path, "d4_crag_path", cfg.d4_crag_path, PINNED_RAG_CRAG_PATH)
    return errors


def _normalized_float_list(values: list[float]) -> list[float]:
    return [float(f"{float(v):.2f}") for v in values]


def _expect_equal(
    errors: list[dict[str, Any]],
    path: Path,
    field: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual == expected:
        return
    errors.append(
        {
            "file": str(path),
            "field": field,
            "expected": expected,
            "actual": actual,
            "message": f"{field} drift in {path.name}: expected {expected!r}, got {actual!r}",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify scenario configs against pinned benchmark values")
    parser.add_argument("--config-dir", default="configs/scenarios")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_scenario_config_dir(Path(args.config_dir))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print(f"pin verification passed: {summary['files_checked']} files checked")
        else:
            print(f"pin verification failed: {summary['error_count']} drift(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
