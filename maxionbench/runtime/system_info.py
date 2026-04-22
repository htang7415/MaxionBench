"""Runtime and hardware summary collection for run metadata."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any


def collect_system_info() -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "apple_silicon_model": _apple_silicon_model(),
        "macos_version": _macos_version(),
        "docker_version": _docker_version(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count() or 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "container_runtime_hint": _detect_container_runtime(),
        "total_memory_bytes": _total_memory_bytes(),
        "gpu_count": _gpu_count(),
    }
    return info


def _total_memory_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return int(pages * page_size)
    except Exception:
        pass
    return 0


def _gpu_count() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode != 0:
            return 0
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        return len(lines)
    except Exception:
        return 0


def _detect_container_runtime() -> str | None:
    if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
        return "apptainer"
    if os.environ.get("ENROOT_ROOTFS"):
        return "enroot"
    if _path_exists("/.dockerenv"):
        return "docker"
    return None


def _path_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _apple_silicon_model() -> str | None:
    if platform.system() != "Darwin":
        return None
    model = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if model:
        return model
    processor = str(platform.processor()).strip()
    return processor or None


def _macos_version() -> str | None:
    if platform.system() != "Darwin":
        return None
    version = str(platform.mac_ver()[0]).strip()
    return version or None


def _docker_version() -> str | None:
    return _run_command(["docker", "--version"])


def _run_command(argv: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    output = str(proc.stdout or "").strip()
    return output or None
