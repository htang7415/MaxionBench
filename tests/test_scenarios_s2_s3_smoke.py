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


def test_calibrate_d3_scenario_smoke(tmp_path: Path) -> None:
    out_params = tmp_path / "d3_params.yaml"
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "calibrate_d3",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3",
        "seed": 5,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "calib-run"),
        "output_d3_params_path": str(out_params),
        "quality_target": 0.8,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 1200,
        "num_queries": 40,
        "top_k": 10,
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 5,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    assert (out_dir / "results.parquet").exists()
    assert out_params.exists()


def test_s2_filtered_ann_scenario_smoke(tmp_path: Path) -> None:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3",
        "seed": 8,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s2-run"),
        "quality_target": 0.8,
        "clients_read": 4,
        "clients_write": 0,
        "clients_grid": [4],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 600,
        "num_queries": 25,
        "top_k": 10,
        "s2_selectivities": [0.01, 0.5],
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 8,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    df = pd.read_parquet(out_dir / "results.parquet")
    assert len(df) == 3  # includes unfiltered baseline + 2 selectivities
    payload = json.loads(df.iloc[0]["search_params_json"])
    assert "selectivity" in payload


def test_s3_and_s3b_smoke(tmp_path: Path) -> None:
    common = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3",
        "seed": 9,
        "repeats": 1,
        "no_retry": True,
        "quality_target": 0.8,
        "clients_read": 8,
        "clients_write": 2,
        "clients_grid": [8],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 120.0,
        "vector_dim": 16,
        "num_vectors": 500,
        "num_queries": 30,
        "top_k": 10,
        "lambda_req_s": 200.0,
        "s3_read_rate": 160.0,
        "s3_insert_rate": 20.0,
        "s3_update_rate": 10.0,
        "s3_delete_rate": 10.0,
        "maintenance_interval_s": 15.0,
        "s3_max_events": 300,
        "d3_k_clusters": 48,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 9,
    }
    s3_cfg = dict(common)
    s3_cfg["scenario"] = "s3_churn_smooth"
    s3_cfg["output_dir"] = str(tmp_path / "s3-run")
    s3_out = _run_cfg(tmp_path, s3_cfg)
    s3_df = pd.read_parquet(s3_out / "results.parquet")
    assert len(s3_df) == 1

    s3b_cfg = dict(common)
    s3b_cfg["scenario"] = "s3b_churn_bursty"
    s3b_cfg["output_dir"] = str(tmp_path / "s3b-run")
    s3b_cfg["s3b_on_s"] = 10.0
    s3b_cfg["s3b_off_s"] = 20.0
    s3b_out = _run_cfg(tmp_path, s3b_cfg)
    s3b_df = pd.read_parquet(s3b_out / "results.parquet")
    assert len(s3b_df) == 1
