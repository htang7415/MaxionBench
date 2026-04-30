from __future__ import annotations

from maxionbench.runtime.system_info import collect_system_info


def test_collect_system_info_has_expected_keys() -> None:
    info = collect_system_info()
    required = {
        "hostname",
        "platform",
        "apple_silicon_model",
        "macos_version",
        "docker_version",
        "python_version",
        "cpu_count_logical",
        "slurm_job_id",
        "slurm_array_task_id",
        "container_runtime_hint",
        "total_memory_bytes",
        "gpu_count",
    }
    assert required.issubset(set(info.keys()))
    assert isinstance(info["cpu_count_logical"], int)
    assert isinstance(info["total_memory_bytes"], int)
    assert isinstance(info["gpu_count"], int)
