from __future__ import annotations

import pytest

from maxionbench.schemas.result_schema import RunMetadata


def _base_metadata(*, hardware_runtime: dict[str, object] | None) -> RunMetadata:
    return RunMetadata(
        run_id="run-1",
        timestamp_utc="2026-03-05T00:00:00+00:00",
        engine="mock",
        engine_version="0.1.0",
        scenario="s1_single_hop",
        dataset_bundle="D4",
        dataset_hash="portable-d4-v1",
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
        rtt_baseline_request_profile="healthcheck_plus_query_topk1_zero_vector",
        sla_threshold_ms=50.0,
        rhu_weights={"w_c": 0.25, "w_g": 0.25, "w_r": 0.25, "w_d": 0.25},
        config_fingerprint="cfg-1",
        repeats=1,
        no_retry=True,
        hardware_runtime=hardware_runtime,
        dataset_cache_checksums=[],
    )


def test_run_metadata_validate_accepts_valid_hardware_runtime_mapping() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
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
    with pytest.raises(ValueError, match="hardware_runtime must be provided as a dict"):
        metadata.validate()


def test_run_metadata_validate_requires_hardware_runtime_dict() -> None:
    class _CustomMapping(dict):
        pass

    metadata = _base_metadata(hardware_runtime=_CustomMapping())
    with pytest.raises(ValueError, match="hardware_runtime must be provided as a dict"):
        metadata.validate()


def test_run_metadata_validate_requires_hardware_runtime_keys() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
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


def test_run_metadata_validate_rejects_blank_gpu_omission_reason_when_omitted() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    metadata = _base_metadata(hardware_runtime=payload)
    metadata = RunMetadata(**{**metadata.__dict__, "gpu_tracks_omitted": True, "gpu_tracks_omission_reason": "  "})
    with pytest.raises(ValueError, match="gpu_tracks_omission_reason"):
        metadata.validate()


def test_run_metadata_validate_rejects_unknown_rtt_baseline_request_profile() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    metadata = _base_metadata(hardware_runtime=payload)
    metadata = RunMetadata(
        **{
            **metadata.__dict__,
            "rtt_baseline_request_profile": "healthcheck_only",
        }
    )
    with pytest.raises(ValueError, match="rtt_baseline_request_profile"):
        metadata.validate()


def test_run_metadata_validate_accepts_dataset_cache_checksum_provenance() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    h = "a" * 64
    metadata = _base_metadata(hardware_runtime=payload)
    metadata = RunMetadata(
        **{
            **metadata.__dict__,
            "dataset_cache_checksums": [
                {
                    "path_key": "dataset_path",
                    "resolved_path": "/tmp/d1.hdf5",
                    "source": "config key dataset_path_sha256",
                    "expected_sha256": h,
                    "actual_sha256": h,
                }
            ],
        }
    )
    metadata.validate()


def test_run_metadata_validate_rejects_invalid_dataset_cache_checksum_provenance() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    metadata = _base_metadata(hardware_runtime=payload)
    metadata = RunMetadata(
        **{
            **metadata.__dict__,
            "dataset_cache_checksums": [
                {
                    "path_key": "dataset_path",
                    "resolved_path": "/tmp/d1.hdf5",
                    "source": "config key dataset_path_sha256",
                    "expected_sha256": "a" * 64,
                    "actual_sha256": "b" * 64,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="expected_sha256 must equal actual_sha256"):
        metadata.validate()


def test_run_metadata_validate_rejects_unknown_budget_level() -> None:
    payload = {
        "hostname": "test-host",
        "platform": "linux",
        "apple_silicon_model": None,
        "macos_version": None,
        "docker_version": None,
        "python_version": "3.11.0",
        "cpu_count_logical": 8,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
        "container_runtime_hint": "docker",
        "total_memory_bytes": 1024,
        "gpu_count": 0,
    }
    metadata = RunMetadata(**{**_base_metadata(hardware_runtime=payload).__dict__, "budget_level": "b3"})
    with pytest.raises(ValueError, match="budget_level must be one of b0,b1,b2 when provided"):
        metadata.validate()
