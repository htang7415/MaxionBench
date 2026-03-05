from __future__ import annotations

import pytest

from maxionbench.schemas.result_schema import RunMetadata


def _base_metadata(*, hardware_runtime: dict[str, object] | None) -> RunMetadata:
    return RunMetadata(
        run_id="run-1",
        timestamp_utc="2026-03-05T00:00:00+00:00",
        engine="mock",
        engine_version="0.1.0",
        scenario="s1_ann_frontier",
        dataset_bundle="D1",
        dataset_hash="synthetic-d1-v1",
        seed=42,
        clients_read=1,
        clients_write=0,
        quality_target=0.8,
        ground_truth_source="exact_topk",
        ground_truth_metric="recall_at_10",
        ground_truth_k=10,
        ground_truth_engine="numpy_exact",
        rtt_baseline_ms_p50=1.0,
        rtt_baseline_ms_p99=2.0,
        sla_threshold_ms=50.0,
        rhu_weights={"w_c": 0.25, "w_g": 0.25, "w_r": 0.25, "w_d": 0.25},
        config_fingerprint="cfg-1",
        repeats=1,
        no_retry=True,
        hardware_runtime=hardware_runtime,
    )


def test_run_metadata_validate_accepts_valid_hardware_runtime_mapping() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    metadata = _base_metadata(hardware_runtime=payload)
    metadata.validate()


def test_run_metadata_validate_requires_hardware_runtime_mapping() -> None:
    metadata = _base_metadata(hardware_runtime=None)
    with pytest.raises(ValueError, match="hardware_runtime must be provided as a mapping"):
        metadata.validate()


def test_run_metadata_validate_requires_hardware_runtime_keys() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
    }
    metadata = _base_metadata(hardware_runtime=payload)
    with pytest.raises(ValueError, match="hardware_runtime missing keys"):
        metadata.validate()
