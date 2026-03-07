from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_calibrate_d3_supports_real_dataset_path_npy(tmp_path: Path) -> None:
    vectors_path = tmp_path / "vectors.npy"
    rng = np.random.default_rng(13)
    vectors = rng.standard_normal((140, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)

    out_params = tmp_path / "d3_params_real.yaml"
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "calibrate_d3",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-path",
        "dataset_path": str(vectors_path),
        "seed": 13,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "calib-run-real"),
        "output_d3_params_path": str(out_params),
        "quality_target": 0.8,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 120,
        "num_queries": 40,
        "top_k": 10,
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 13,
    }
    _run_cfg(tmp_path, cfg)
    payload = yaml.safe_load(out_params.read_text(encoding="utf-8"))
    assert payload["calibration_source"] == "real_dataset_path"
    assert int(payload["calibration_vector_count"]) == 120


def test_calibrate_d3_supports_relative_dataset_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = data_dir / "vectors.npy"
    vectors = np.random.default_rng(17).standard_normal((140, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)

    out_params = tmp_path / "d3_params_real_relative.yaml"
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "calibrate_d3",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-relative-path",
        "dataset_path": "data/vectors.npy",
        "seed": 17,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "calib-run-real-relative"),
        "output_d3_params_path": str(out_params),
        "quality_target": 0.8,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 120,
        "num_queries": 40,
        "top_k": 10,
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 17,
    }
    _run_cfg(tmp_path, cfg)
    payload = yaml.safe_load(out_params.read_text(encoding="utf-8"))
    assert payload["calibration_source"] == "real_dataset_path"
    assert int(payload["calibration_vector_count"]) == 120


def test_calibrate_d3_supports_env_placeholder_dataset_path(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    vectors_path = tmp_path / "vectors_env.npy"
    vectors = np.random.default_rng(23).standard_normal((140, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)
    monkeypatch.setenv("MAXIONBENCH_D3_DATASET_PATH", str(vectors_path))

    out_params = tmp_path / "d3_params_real_env.yaml"
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "calibrate_d3",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-env-path",
        "dataset_path": "${MAXIONBENCH_D3_DATASET_PATH}",
        "calibration_require_real_data": True,
        "seed": 23,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "calib-run-real-env"),
        "output_d3_params_path": str(out_params),
        "quality_target": 0.8,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 120,
        "num_queries": 40,
        "top_k": 10,
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 23,
    }
    _run_cfg(tmp_path, cfg)
    payload = yaml.safe_load(out_params.read_text(encoding="utf-8"))
    assert payload["calibration_source"] == "real_dataset_path"
    assert int(payload["calibration_vector_count"]) == 120


def test_calibrate_d3_requires_real_dataset_when_flag_enabled(tmp_path: Path) -> None:
    out_params = tmp_path / "d3_params_missing_real.yaml"
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "calibrate_d3",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-missing-real-path",
        "calibration_require_real_data": True,
        "seed": 29,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "calib-run-missing-real"),
        "output_d3_params_path": str(out_params),
        "quality_target": 0.8,
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 80.0,
        "vector_dim": 16,
        "num_vectors": 120,
        "num_queries": 40,
        "top_k": 10,
        "d3_k_clusters": 64,
        "d3_num_tenants": 20,
        "d3_num_acl_buckets": 8,
        "d3_num_time_buckets": 12,
        "d3_beta_tenant": 0.75,
        "d3_beta_acl": 0.7,
        "d3_beta_time": 0.65,
        "d3_seed": 29,
    }
    with pytest.raises(ValueError, match="requires a real D3 dataset_path"):
        _run_cfg(tmp_path, cfg)


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


def test_s1_d3_supports_relative_dataset_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = data_dir / "s1_d3_vectors.npy"
    rng = np.random.default_rng(41)
    vectors = rng.standard_normal((700, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-s1-relative",
        "dataset_path": "data/s1_d3_vectors.npy",
        "seed": 41,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s1-d3-relative"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 600,
        "num_queries": 30,
        "top_k": 10,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["ground_truth_source"] == "exact_topk"


def test_s1_d3_rejects_dataset_path_dimension_mismatch(tmp_path: Path) -> None:
    vectors_path = tmp_path / "s1_d3_bad_dim.npy"
    np.save(vectors_path, np.random.default_rng(43).standard_normal((640, 12), dtype=np.float32))
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-s1-bad-dim",
        "dataset_path": str(vectors_path),
        "seed": 43,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s1-d3-bad-dim"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 600,
        "num_queries": 30,
        "top_k": 10,
    }
    with pytest.raises(ValueError, match="dimension mismatch"):
        _run_cfg(tmp_path, cfg)


def test_s2_filtered_ann_supports_real_dataset_path_npy(tmp_path: Path) -> None:
    vectors_path = tmp_path / "s2_vectors.npy"
    rng = np.random.default_rng(21)
    vectors = rng.standard_normal((720, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)
    checksum = hashlib.sha256(vectors_path.read_bytes()).hexdigest()

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-s2",
        "dataset_path": str(vectors_path),
        "dataset_path_sha256": checksum,
        "seed": 21,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s2-run-real"),
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
        "d3_seed": 21,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    df = pd.read_parquet(out_dir / "results.parquet")
    assert len(df) == 3
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    checks = metadata["dataset_cache_checksums"]
    assert any(str(item.get("path_key")) == "dataset_path" for item in checks)


def test_s2_filtered_ann_supports_relative_dataset_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = data_dir / "s2_vectors.npy"
    rng = np.random.default_rng(31)
    vectors = rng.standard_normal((720, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-s2-relative",
        "dataset_path": "data/s2_vectors.npy",
        "seed": 31,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s2-run-real-relative"),
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
        "d3_seed": 31,
    }
    out_dir = _run_cfg(tmp_path, cfg)
    df = pd.read_parquet(out_dir / "results.parquet")
    assert len(df) == 3


def test_s2_filtered_ann_rejects_dataset_path_dimension_mismatch(tmp_path: Path) -> None:
    vectors_path = tmp_path / "s2_vectors_bad_dim.npy"
    np.save(vectors_path, np.random.default_rng(23).standard_normal((640, 12), dtype=np.float32))

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-s2-bad-dim",
        "dataset_path": str(vectors_path),
        "seed": 23,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s2-run-real-bad-dim"),
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
        "d3_seed": 23,
    }
    with pytest.raises(ValueError, match="dimension mismatch"):
        _run_cfg(tmp_path, cfg)


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
    s1_cfg = dict(common)
    s1_cfg["scenario"] = "s1_ann_frontier"
    s1_cfg["output_dir"] = str(tmp_path / "s1-d3-baseline")
    s1_cfg["clients_write"] = 0
    s1_cfg["quality_targets"] = [s1_cfg["quality_target"]]
    _run_cfg(tmp_path, s1_cfg)

    s3_cfg = dict(common)
    s3_cfg["scenario"] = "s3_churn_smooth"
    s3_cfg["output_dir"] = str(tmp_path / "s3-run")
    s3_out = _run_cfg(tmp_path, s3_cfg)
    s3_df = pd.read_parquet(s3_out / "results.parquet")
    assert len(s3_df) == 1
    s3_payload = json.loads(str(s3_df.iloc[0]["search_params_json"]))
    assert float(s3_payload["s1_baseline_p99_ms"]) > 0.0
    assert int(s3_payload["s1_baseline_match_rows"]) >= 1
    assert float(s3_payload["p99_inflation_vs_s1_baseline"]) >= 0.0
    assert str(s3_payload["burst_clock_anchor"]) == "measurement_start"

    s3b_cfg = dict(common)
    s3b_cfg["scenario"] = "s3b_churn_bursty"
    s3b_cfg["output_dir"] = str(tmp_path / "s3b-run")
    s3b_cfg["s3b_on_s"] = 10.0
    s3b_cfg["s3b_off_s"] = 20.0
    s3b_out = _run_cfg(tmp_path, s3b_cfg)
    s3b_df = pd.read_parquet(s3b_out / "results.parquet")
    assert len(s3b_df) == 1
    s3b_payload = json.loads(str(s3b_df.iloc[0]["search_params_json"]))
    assert float(s3b_payload["s1_baseline_p99_ms"]) > 0.0
    assert int(s3b_payload["s1_baseline_match_rows"]) >= 1
    assert float(s3b_payload["p99_inflation_vs_s1_baseline"]) >= 0.0
    assert str(s3b_payload["burst_clock_anchor"]) == "measurement_start"
    assert str(s3b_payload["mode"]) == "s3_bursty"
    assert float(s3b_payload["burst_on_s"]) == 10.0
    assert float(s3b_payload["burst_off_s"]) == 20.0
    assert float(s3b_payload["burst_cycle_s"]) == 30.0
    assert float(s3b_payload["burst_on_write_mult"]) == 8.0
    assert float(s3b_payload["burst_off_write_mult"]) == 0.25
    assert float(s3b_payload["burst_write_multiplier_normalizer"]) == pytest.approx(85.0 / 30.0)
    assert float(s3b_payload["burst_effective_mean_write_mult"]) == pytest.approx(1.0)


def test_s3_rejects_dataset_path_with_too_few_vectors(tmp_path: Path) -> None:
    vectors_path = tmp_path / "s3_vectors_short.npy"
    rng = np.random.default_rng(29)
    vectors = rng.standard_normal((200, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    np.save(vectors_path, vectors)

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s3_churn_smooth",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3-real-s3-short",
        "dataset_path": str(vectors_path),
        "seed": 29,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s3-run-real-short"),
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
        "d3_seed": 29,
    }
    with pytest.raises(ValueError, match="fewer than requested"):
        _run_cfg(tmp_path, cfg)


def test_s3_requires_matched_s1_baseline(tmp_path: Path) -> None:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s3_churn_smooth",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3",
        "seed": 19,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s3-run"),
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
    with pytest.raises(RuntimeError, match="matched S1 baseline"):
        _run_cfg(tmp_path, cfg)


def test_s2_rejects_non_paper_d3_params_file(tmp_path: Path) -> None:
    bad_params = tmp_path / "d3_params_bad.yaml"
    bad_payload = {
        "k_clusters": 4096,
        "num_tenants": 100,
        "num_acl_buckets": 16,
        "num_time_buckets": 52,
        "beta_tenant": 0.95,
        "beta_acl": 0.95,
        "beta_time": 0.90,
        "seed": 42,
        "calibration_eval": {
            "test_a_median_concentration": 0.547,
            "test_b_cluster_spread": 42.775,
            "p99_ratio_1pct_to_50pct": 1.19,
            "recall_gap_50_minus_1": -0.79,
            "trivial": True,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    bad_params.write_text(yaml.safe_dump(bad_payload, sort_keys=True), encoding="utf-8")

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
        "d3_seed": 8,
    }
    cfg_path = tmp_path / "s2_reject_bad_d3.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="not paper-ready"):
        run_from_config(cfg_path, cli_overrides={"d3_params": str(bad_params)})


def test_s2_allows_non_paper_d3_params_when_explicitly_opted_in(tmp_path: Path) -> None:
    bad_params = tmp_path / "d3_params_bad_opt_in.yaml"
    bad_payload = {
        "k_clusters": 4096,
        "num_tenants": 100,
        "num_acl_buckets": 16,
        "num_time_buckets": 52,
        "beta_tenant": 0.95,
        "beta_acl": 0.95,
        "beta_time": 0.90,
        "seed": 42,
        "calibration_eval": {
            "test_a_median_concentration": 0.547,
            "test_b_cluster_spread": 42.775,
            "p99_ratio_1pct_to_50pct": 1.19,
            "recall_gap_50_minus_1": -0.79,
            "trivial": True,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    bad_params.write_text(yaml.safe_dump(bad_payload, sort_keys=True), encoding="utf-8")

    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s2_filtered_ann",
        "dataset_bundle": "D3",
        "dataset_hash": "synthetic-d3",
        "seed": 8,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "s2-run-opt-in"),
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
        "d3_seed": 8,
        "allow_unverified_d3_params": True,
    }
    cfg_path = tmp_path / "s2_allow_bad_d3.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    out_dir = run_from_config(cfg_path, cli_overrides={"d3_params": str(bad_params)})
    assert (out_dir / "results.parquet").exists()
