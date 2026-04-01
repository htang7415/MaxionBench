from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from maxionbench.adapters.mock import MockAdapter
from maxionbench.orchestration import runner as runner_mod
from maxionbench.orchestration.runner import _gpu_count_for_cfg, _resolve_d3_params, run_from_config
from maxionbench.orchestration.config_schema import RunConfig, expand_env_placeholders, load_run_config
from maxionbench.scenarios.s1_ann_frontier import S1Data
from maxionbench.schemas.result_schema import read_run_status
from maxionbench.tools.validate_outputs import validate_run_directory


def test_runner_end_to_end(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 7,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 20,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 200,
        "num_queries": 20,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "config_resolved.yaml").exists()
    assert (out_dir / "run_status.json").exists()
    assert (out_dir / "logs" / "runner.log").exists()

    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    assert set(
        [
            "recall_at_10",
            "p99_ms",
            "qps",
            "resource_cpu_vcpu",
            "resource_gpu_count",
            "resource_ram_gib",
            "resource_disk_tb",
            "rhu_rate",
        ]
    ).issubset(frame.columns)
    row = frame.iloc[0]
    assert float(row["resource_cpu_vcpu"]) >= 1.0
    assert float(row["resource_ram_gib"]) >= 0.0
    assert float(row["resource_disk_tb"]) >= 0.0
    assert float(row["rhu_rate"]) > 0.0

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["resource_profile"].keys()) == {"cpu_vcpu", "gpu_count", "ram_gib", "disk_tb", "rhu_rate"}
    assert set(metadata["rhu_references"].keys()) == {"c_ref_vcpu", "g_ref_gpu", "r_ref_gib", "d_ref_tb"}
    assert metadata["ground_truth_source"] == "exact_topk"
    assert metadata["ground_truth_metric"] == "recall_at_10"
    assert int(metadata["ground_truth_k"]) == 10
    assert metadata["ground_truth_engine"] == "numpy_exact"
    log_lines = [line for line in (out_dir / "logs" / "runner.log").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(log_lines) == len(frame)
    event = json.loads(log_lines[0])
    assert event["run_id"] == row["run_id"]
    assert event["config_fingerprint"] == metadata["config_fingerprint"]
    assert event["scenario"] == row["scenario"]
    assert event["engine"] == row["engine"]
    run_status = read_run_status(out_dir / "run_status.json")
    assert run_status["status"] == "success"
    assert int(run_status["exit_code"]) == 0

    summary = validate_run_directory(out_dir)
    assert summary["rows"] == 1


def test_create_benchmark_adapter_applies_index_params_before_create(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _FakeAdapter:
        def set_index_params(self, params):  # type: ignore[no-untyped-def]
            calls.append(("set_index_params", dict(params)))

        def create(self, collection, dimension, metric="ip"):  # type: ignore[no-untyped-def]
            calls.append(("create", collection, dimension, metric))

    fake_adapter = _FakeAdapter()
    monkeypatch.setattr(runner_mod, "create_adapter", lambda engine, **options: fake_adapter)

    cfg = RunConfig(
        engine="pgvector",
        adapter_options={"dsn": "postgresql://postgres:postgres@127.0.0.1:5432/postgres"},
        index_params={"lists": 256},
        no_retry=True,
    )

    adapter = runner_mod._create_benchmark_adapter(cfg=cfg)

    assert adapter is fake_adapter
    assert calls == [
        ("set_index_params", {"lists": 256}),
        ("create", "maxionbench", cfg.vector_dim, "ip"),
    ]


def test_create_benchmark_adapter_normalizes_dataset_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _FakeAdapter:
        def set_index_params(self, params):  # type: ignore[no-untyped-def]
            calls.append(("set_index_params", dict(params)))

        def create(self, collection, dimension, metric="ip"):  # type: ignore[no-untyped-def]
            calls.append(("create", collection, dimension, metric))

    fake_adapter = _FakeAdapter()
    monkeypatch.setattr(runner_mod, "create_adapter", lambda engine, **options: fake_adapter)

    cfg = RunConfig(engine="mock", no_retry=True)

    adapter = runner_mod._create_benchmark_adapter(cfg=cfg, metric="angular")

    assert adapter is fake_adapter
    assert calls == [("create", "maxionbench", cfg.vector_dim, "cos")]


def test_run_s1_rows_reuses_loaded_data_adapter_across_sweeps_clients_and_repeats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _SpyAdapter(MockAdapter):
        instances: list["_SpyAdapter"] = []

        def __init__(self) -> None:
            super().__init__()
            self.flush_calls = 0
            self.drop_calls = 0
            _SpyAdapter.instances.append(self)

        def flush_or_commit(self) -> None:
            self.flush_calls += 1
            super().flush_or_commit()

        def drop(self, collection: str) -> None:
            self.drop_calls += 1
            super().drop(collection)

    s1_data = S1Data(
        ids=["doc-0", "doc-1", "doc-2", "doc-3"],
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            dtype="float32",
        ),
        queries=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        ground_truth_ids=[["doc-0"], ["doc-1"]],
        metric="ip",
    )
    monkeypatch.setattr(runner_mod, "_maybe_load_s1_data", lambda cfg, config_path: s1_data)
    monkeypatch.setattr(runner_mod, "create_adapter", lambda engine, **options: _SpyAdapter())
    monkeypatch.setattr(
        runner_mod,
        "measure_rpc_baseline",
        lambda request_fn, request_count: {"rtt_baseline_ms_p50": 1.0, "rtt_baseline_ms_p99": 2.0},
    )

    cfg = RunConfig(
        engine="mock",
        scenario="s1_ann_frontier",
        dataset_bundle="D3",
        dataset_hash="processed-d3",
        repeats=2,
        no_retry=True,
        output_dir=str(tmp_path / "run"),
        quality_targets=[0.0],
        clients_grid=[1, 2],
        search_sweep=[{"hnsw_ef": 32}, {"hnsw_ef": 64}],
        rpc_baseline_requests=1,
        warmup_s=0.0,
        steady_state_s=0.01,
        phase_timing_mode="strict",
        phase_max_requests_per_phase=2,
        vector_dim=2,
        num_vectors=4,
        num_queries=2,
        top_k=1,
        sla_threshold_ms=50.0,
    )

    rows, _, _ = runner_mod._run_s1_rows(
        cfg=cfg,
        config_fingerprint="abc123",
        config_path=tmp_path / "cfg.yaml",
    )

    assert len(rows) == 4
    assert len(_SpyAdapter.instances) == 1
    assert _SpyAdapter.instances[0].flush_calls == 1
    assert _SpyAdapter.instances[0].drop_calls == 1


def test_load_run_config_normalizes_pgvector_hnsw_ef_to_ivfflat_probes(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pgvector.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "pgvector",
                "engine_version": "postgres16-pgvector",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D1",
                "dataset_hash": "synthetic-d1-v1",
                "output_dir": str(tmp_path / "run"),
                "no_retry": True,
                "search_sweep": [{"hnsw_ef": 64}, {"hnsw_ef": 128}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)

    assert cfg.search_sweep == [{"ivfflat_probes": 64}, {"ivfflat_probes": 128}]


def test_load_run_config_preserves_pgvector_index_params(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pgvector-index.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "pgvector",
                "engine_version": "postgres16-pgvector",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D2",
                "dataset_hash": "ann-benchmarks-deep-image-96-angular",
                "output_dir": str(tmp_path / "run"),
                "no_retry": True,
                "index_params": {"lists": 4096},
                "search_sweep": [{"hnsw_ef": 64}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)

    assert cfg.index_params == {"lists": 4096}
    assert cfg.search_sweep == [{"ivfflat_probes": 64}]


def test_load_run_config_normalizes_pgvector_hnsw_method_to_hnsw_ef_search(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pgvector-hnsw.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "pgvector",
                "engine_version": "postgres16-pgvector",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D1",
                "dataset_hash": "synthetic-d1-v1",
                "output_dir": str(tmp_path / "run"),
                "no_retry": True,
                "adapter_options": {"index_method": "hnsw"},
                "search_sweep": [{"hnsw_ef": 64}, {"hnsw_ef": 128}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)

    assert cfg.search_sweep == [{"hnsw_ef_search": 64}, {"hnsw_ef_search": 128}]


def test_runner_matched_quality_grid_outputs_one_row_per_target_and_client(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 11,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-grid"),
        "quality_target": 0.8,
        "quality_targets": [0.5, 0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1, 2],
        "search_sweep": [{"hnsw_ef": 32}, {"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 150,
        "num_queries": 10,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config-grid.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 4
    assert set(frame["quality_target"].tolist()) == {0.5, 0.8}
    assert set(frame["clients_read"].tolist()) == {1, 2}


def test_runner_phase_fields_and_strict_mode_cap(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 31,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-phase"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 12,
        "num_vectors": 120,
        "num_queries": 8,
        "top_k": 10,
        "warmup_s": 0.02,
        "steady_state_s": 0.02,
        "phase_timing_mode": "strict",
        "phase_max_requests_per_phase": 12,
    }
    cfg_path = tmp_path / "config-phase.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert float(row["setup_elapsed_s"]) >= 0.0
    assert int(row["warmup_requests"]) > 0
    assert int(row["measure_requests"]) > 0
    assert int(row["measure_requests"]) <= 12
    assert float(row["measure_elapsed_s"]) >= 0.0
    assert float(row["export_elapsed_s"]) >= 0.0


def test_runner_writes_s1_diagnostics_when_no_matched_quality_row_exists(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 17,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-no-rows"),
        "quality_target": 1.1,
        "quality_targets": [1.1],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}, {"hnsw_ef": 64}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 150,
        "num_queries": 10,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config-no-rows.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    with pytest.raises(RuntimeError, match="No feasible matched-quality candidates for s1_ann_frontier"):
        run_from_config(cfg_path, cli_overrides=None)

    out_dir = tmp_path / "run-no-rows"
    diagnostics_path = out_dir / "logs" / "s1_sweep_diagnostics.jsonl"
    summary_path = out_dir / "logs" / "s1_selection_summary.json"
    status_path = out_dir / "run_status.json"

    assert diagnostics_path.exists()
    assert summary_path.exists()
    assert status_path.exists()

    diagnostics_lines = [line for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(diagnostics_lines) == 2
    first_diagnostic = json.loads(diagnostics_lines[0])
    assert first_diagnostic["client_count"] == 1
    assert "quality_targets_met" in first_diagnostic
    assert "error_examples" in first_diagnostic

    selection_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(selection_summary) == 1
    assert selection_summary[0]["selected"] is False
    assert selection_summary[0]["best_available_recall_at_10"] >= 0.0

    run_status = read_run_status(status_path)
    assert run_status["status"] == "failed"
    assert "No feasible matched-quality candidates" in str(run_status["detail"])


def test_validate_outputs_rejects_missing_or_negative_stage_timing(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 43,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-validate"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 12,
        "num_vectors": 120,
        "num_queries": 8,
        "top_k": 10,
        "warmup_s": 0.02,
        "steady_state_s": 0.02,
        "phase_timing_mode": "strict",
        "phase_max_requests_per_phase": 12,
    }
    cfg_path = tmp_path / "config-validate.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(cfg_path, cli_overrides=None)
    frame = pd.read_parquet(out_dir / "results.parquet")

    missing_col = frame.drop(columns=["setup_elapsed_s"])
    missing_col.to_parquet(out_dir / "results.parquet", index=False)
    with pytest.raises(ValueError, match="missing stage timing columns"):
        validate_run_directory(out_dir)

    frame.loc[:, "export_elapsed_s"] = -1.0
    frame.to_parquet(out_dir / "results.parquet", index=False)
    with pytest.raises(ValueError, match="negative values"):
        validate_run_directory(out_dir)


def test_gpu_count_resolution() -> None:
    faiss_gpu_cfg = RunConfig(engine="faiss-gpu", no_retry=True)
    assert _gpu_count_for_cfg(faiss_gpu_cfg) == 1.0

    explicit_cfg = RunConfig(engine="mock", adapter_options={"gpu_count": 2}, no_retry=True)
    assert _gpu_count_for_cfg(explicit_cfg) == 2.0


def test_config_rejects_d4_real_data_without_any_source_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_d4_invalid.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "mock",
                "scenario": "s4_hybrid",
                "dataset_bundle": "D4",
                "dataset_hash": "invalid-d4",
                "no_retry": True,
                "d4_use_real_data": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="requires at least one of processed_dataset_path, d4_beir_root, or d4_crag_path",
    ):
        load_run_config(cfg_path)


def test_run_from_config_rejects_missing_d3_params_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_d3_missing_params.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "mock",
                "engine_version": "0.1.0",
                "scenario": "s2_filtered_ann",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3",
                "seed": 1,
                "repeats": 1,
                "no_retry": True,
                "output_dir": str(tmp_path / "run"),
                "quality_target": 0.8,
                "quality_targets": [0.8],
                "clients_read": 1,
                "clients_write": 0,
                "clients_grid": [1],
                "search_sweep": [{"hnsw_ef": 32}],
                "rpc_baseline_requests": 5,
                "sla_threshold_ms": 80.0,
                "vector_dim": 8,
                "num_vectors": 20,
                "num_queries": 5,
                "top_k": 10,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="d3 params file not found"):
        run_from_config(cfg_path, cli_overrides={"d3_params": "artifacts/calibration/missing.yaml"})


def test_expand_env_placeholders_supports_shell_style_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAXIONBENCH_QDRANT_HOST", raising=False)
    payload = {
        "adapter_options": {
            "host": "${MAXIONBENCH_QDRANT_HOST:-127.0.0.1}",
            "dsn": "postgresql://postgres:postgres@${MAXIONBENCH_PGVECTOR_HOST:-127.0.0.1}:${MAXIONBENCH_PGVECTOR_PORT:-5432}/postgres",
            "inproc_uri": "${MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI:-}",
        }
    }

    expanded = expand_env_placeholders(payload)
    assert expanded["adapter_options"]["host"] == "127.0.0.1"
    assert expanded["adapter_options"]["dsn"] == "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    assert expanded["adapter_options"]["inproc_uri"] is None


def test_load_run_config_resolves_docker_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAXIONBENCH_QDRANT_HOST", "qdrant")
    cfg_path = tmp_path / "cfg_env.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "engine": "qdrant",
                "engine_version": "server",
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D1",
                "dataset_hash": "synthetic-d1-v1",
                "no_retry": True,
                "adapter_options": {
                    "host": "${MAXIONBENCH_QDRANT_HOST:-127.0.0.1}",
                    "port": "${MAXIONBENCH_QDRANT_PORT:-6333}",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)
    assert cfg.adapter_options["host"] == "qdrant"
    assert cfg.adapter_options["port"] == "6333"


def test_resolve_d3_params_reuses_calibrated_affinities_but_preserves_50m_k_pin(tmp_path: Path) -> None:
    d3_params_path = tmp_path / "d3_params.yaml"
    payload = {
        "k_clusters": 4096,
        "num_tenants": 100,
        "num_acl_buckets": 16,
        "num_time_buckets": 52,
        "beta_tenant": 0.91,
        "beta_acl": 0.83,
        "beta_time": 0.79,
        "seed": 123,
        "calibration_eval": {
            "test_a_median_concentration": 0.61,
            "test_b_cluster_spread": 20.0,
            "p99_ratio_1pct_to_50pct": 2.3,
            "recall_gap_50_minus_1": 0.08,
            "trivial": False,
        },
        "calibration_vector_count": 10_000_000,
        "calibration_source": "real_dataset_path",
    }
    d3_params_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    cfg = RunConfig(
        engine="mock",
        scenario="s2_filtered_ann",
        dataset_bundle="D3",
        dataset_hash="synthetic-d3-50m",
        num_vectors=50_000_000,
        d3_k_clusters=8192,
        allow_unverified_d3_params=True,
        no_retry=True,
    )
    resolved = _resolve_d3_params(cfg, str(d3_params_path))
    assert resolved.k_clusters == 8192
    assert resolved.beta_tenant == pytest.approx(0.91)
    assert resolved.beta_acl == pytest.approx(0.83)
    assert resolved.beta_time == pytest.approx(0.79)
    assert resolved.seed == 123


def test_resolve_d3_params_requires_calibration_file_for_strict_d3_runs() -> None:
    cfg = RunConfig(
        engine="mock",
        scenario="s2_filtered_ann",
        dataset_bundle="D3",
        dataset_hash="synthetic-d3-10m",
        phase_timing_mode="strict",
        no_retry=True,
    )
    with pytest.raises(ValueError, match="d3 params are required for strict D3 robustness scenarios"):
        _resolve_d3_params(cfg, None)


def test_runner_enforce_readiness_allows_mock_without_matrix(tmp_path: Path) -> None:
    config = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 83,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-readiness-mock"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 12,
        "num_vectors": 120,
        "num_queries": 8,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config-readiness-mock.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    out_dir = run_from_config(
        cfg_path,
        cli_overrides={
            "enforce_readiness": True,
            "conformance_matrix": str(tmp_path / "missing_conformance.csv"),
            "behavior_dir": str(tmp_path / "missing_behavior"),
        },
    )
    assert (out_dir / "results.parquet").exists()
    resolved = yaml.safe_load((out_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert isinstance(resolved, dict)
    readiness = resolved.get("readiness")
    assert isinstance(readiness, dict)
    assert readiness.get("enforced") is True


def test_runner_enforce_readiness_blocks_real_engine_when_not_ready(tmp_path: Path) -> None:
    config = {
        "engine": "qdrant",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 89,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run-readiness-qdrant"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 5,
        "sla_threshold_ms": 50.0,
        "vector_dim": 12,
        "num_vectors": 120,
        "num_queries": 8,
        "top_k": 10,
    }
    cfg_path = tmp_path / "config-readiness-qdrant.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    conformance_matrix = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "mock", "status": "pass"}]).to_csv(conformance_matrix, index=False)
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    with pytest.raises(RuntimeError, match="pre-run readiness gate failed"):
        run_from_config(
            cfg_path,
            cli_overrides={
                "enforce_readiness": True,
                "conformance_matrix": str(conformance_matrix),
                "behavior_dir": str(behavior_dir),
            },
        )
