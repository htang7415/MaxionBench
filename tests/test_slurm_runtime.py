from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from maxionbench.orchestration.slurm import preflight as preflight_mod
from maxionbench.orchestration.slurm.preflight import evaluate_preflight
from maxionbench.runtime.ports import allocate_named_ports, allocate_port_range


def test_allocate_port_range_and_named_ports() -> None:
    ports = allocate_port_range(count=3, base=25000, span=1000, offset=10)
    assert len(ports) == 3
    assert ports[1] == ports[0] + 1

    named = allocate_named_ports(["a", "b", "a"], base=26000, span=1000)
    assert set(named.keys()) == {"a", "b"}
    assert named["b"] == named["a"] + 1


def test_preflight_uses_manifest_when_available(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "D3"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["fallback_config"] == "configs/scenarios/s2_filtered_ann.yaml"


def test_preflight_estimate_without_manifest(tmp_path: Path) -> None:
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_unknown.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["dataset_bundle"] == "UNKNOWN"
    assert summary["manifest_ok"] is True
    assert summary["manifest_error"] is None
    assert summary["dataset_bytes"] > 0
    assert summary["engine_bytes"] > 0
    assert summary["temp_bytes"] > 0


def test_preflight_verifies_dataset_cache_checksum_when_provided(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample.bin"
    dataset_file.write_bytes(b"cache-check")
    checksum = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": checksum,
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_ok.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is True
    assert int(summary["integrity_error_count"]) == 0
    assert int(summary["integrity_checked_files"]) == 1


def test_preflight_fails_when_dataset_cache_checksum_mismatches(tmp_path: Path) -> None:
    dataset_file = tmp_path / "sample_bad.bin"
    dataset_file.write_bytes(b"cache-check-bad")
    cfg = {
        "dataset_bundle": "UNKNOWN",
        "dataset_path": str(dataset_file),
        "dataset_path_sha256": ("0" * 64),
        "num_vectors": 100,
        "vector_dim": 8,
    }
    cfg_path = tmp_path / "cfg_checksum_bad.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["integrity_ok"] is False
    assert int(summary["integrity_error_count"]) >= 1
    assert summary["ok"] is False
    assert any("sha256 mismatch" in msg for msg in summary["integrity_errors"])


def test_preflight_fails_when_known_bundle_manifest_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_missing_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(preflight_mod, "_load_manifest", lambda _: None)
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "missing manifest" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_preflight_fails_when_known_bundle_manifest_has_invalid_size_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    cfg = {
        "dataset_bundle": "D3",
        "num_vectors": 1000,
        "vector_dim": 16,
    }
    cfg_path = tmp_path / "cfg_known_bundle_bad_manifest.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)

    monkeypatch.setattr(
        preflight_mod,
        "_load_manifest",
        lambda _: {
            "dataset_bundle": "D3",
            "approx_bytes_dataset": 0,
            "approx_bytes_engine": 100,
            "approx_bytes_temp": 10,
        },
    )
    summary = evaluate_preflight(config_path=cfg_path, tmpdir=tmp_path, safety_factor=1.8)
    assert summary["manifest_ok"] is False
    assert "must be > 0" in str(summary["manifest_error"])
    assert summary["ok"] is False


def test_slurm_common_runs_pre_run_gate_before_runner() -> None:
    text = Path("maxionbench/orchestration/slurm/common.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_PRE_RUN_GATE" in text
    assert "MAXIONBENCH_ALLOW_GPU_UNAVAILABLE" in text
    assert "MAXIONBENCH_CONFORMANCE_MATRIX" in text
    assert "MAXIONBENCH_OUTPUT_ROOT" in text
    assert "MAXIONBENCH_CONTAINER_RUNTIME" in text
    assert "MAXIONBENCH_CONTAINER_IMAGE" in text
    assert "MAXIONBENCH_CONTAINER_BIND" in text
    assert "MAXIONBENCH_HF_CACHE_DIR" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "mb_source_dataset_env()" in text
    assert "apptainer exec" in text
    assert "mb_python()" in text
    gate_marker = "pre-run-gate"
    runner_marker = "python -m maxionbench.orchestration.runner"
    assert gate_marker in text
    assert runner_marker in text
    assert text.index(gate_marker) < text.index(runner_marker)


def test_cpu_array_includes_d3_matched_s1_baseline_config() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "configs/scenarios/s1_ann_frontier_d3.yaml" in text


def test_cpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text


def test_cpu_array_supports_skip_s6_env_flag() -> None:
    text = Path("maxionbench/orchestration/slurm/cpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SKIP_S6" in text
    assert "s6_fusion.yaml" in text
    assert "skipping S6 task index" in text


def test_gpu_array_supports_partial_scenario_dir_override_fallback() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_SCENARIO_CONFIG_DIR" in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/$(basename "${DEFAULT_CONFIG_PATH}")"' in text
    assert 'if [[ -f "$(mb_resolve_config "${CANDIDATE_CONFIG_PATH}")" ]]; then' in text
    assert 'CONFIG_PATH="${DEFAULT_CONFIG_PATH}"' in text


def test_gpu_array_explicitly_lists_track_b_and_track_c_entries() -> None:
    text = Path("maxionbench/orchestration/slurm/gpu_array.sh").read_text(encoding="utf-8")
    assert "s1_ann_frontier_track_b_gpu.yaml" in text
    assert "s5_rerank_track_c_gpu.yaml" in text


def test_calibrate_d3_supports_scenario_dir_override_with_explicit_override_precedence() -> None:
    text = Path("maxionbench/orchestration/slurm/calibrate_d3.sh").read_text(encoding="utf-8")
    assert 'CONFIG_PATH="${MAXIONBENCH_CALIBRATE_CONFIG:-configs/scenarios/calibrate_d3.yaml}"' in text
    assert 'if [[ -z "${MAXIONBENCH_CALIBRATE_CONFIG:-}" ]]; then' in text
    assert 'SCENARIO_CONFIG_DIR="${MAXIONBENCH_SCENARIO_CONFIG_DIR:-}"' in text
    assert 'CANDIDATE_CONFIG_PATH="${SCENARIO_CONFIG_DIR}/calibrate_d3.yaml"' in text
    assert 'if [[ ! -f "$(mb_resolve_config "${CONFIG_PATH}")" ]]; then' in text
    assert "mb_source_dataset_env" in text


def test_prefetch_datasets_script_exists_and_uses_prefetch_helper() -> None:
    text = Path("maxionbench/orchestration/slurm/prefetch_datasets.sh").read_text(encoding="utf-8")
    assert "dataset_prefetch" in text
    assert "MAXIONBENCH_DATASET_ENV_SH" in text
    assert "mb_source_dataset_env" in text
