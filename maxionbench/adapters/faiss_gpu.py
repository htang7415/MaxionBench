"""FAISS GPU adapter."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from .faiss_cpu import FaissCpuAdapter

_FAISS_GPU_PROBE_SCRIPT = """
import sys

import numpy as np

import faiss

device_id = int(sys.argv[1])
resources = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(4)
index.add(np.zeros((1, 4), dtype=np.float32))
faiss.index_cpu_to_gpu(resources, device_id, index)
"""


class FaissGpuAdapter(FaissCpuAdapter):
    """FAISS GPU adapter based on FAISS CPU index transfer."""

    _runtime_probe_cache: dict[tuple[str, int], str] = {}

    def __init__(self, device_id: int = 0) -> None:
        super().__init__()
        if not hasattr(self._faiss, "StandardGpuResources"):
            raise RuntimeError(
                "Installed FAISS does not include GPU support. Install a GPU-enabled FAISS wheel such as "
                "`faiss-gpu-cu12` for FaissGpuAdapter."
            )
        self._device_id = int(device_id)
        self._gpu_res = self._faiss.StandardGpuResources()
        self._ensure_runtime_compatible()

    def _finalize_index(self, cpu_index: Any) -> Any:
        return self._faiss.index_cpu_to_gpu(self._gpu_res, self._device_id, cpu_index)

    def _ensure_runtime_compatible(self) -> None:
        cache_key = (str(getattr(self._faiss, "__version__", "unknown")), self._device_id)
        cached_error = self._runtime_probe_cache.get(cache_key)
        if cached_error is not None:
            if cached_error:
                raise RuntimeError(cached_error)
            return

        probe = subprocess.run(
            [sys.executable, "-c", _FAISS_GPU_PROBE_SCRIPT, str(self._device_id)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            self._runtime_probe_cache[cache_key] = ""
            return

        detail = (probe.stderr or probe.stdout).strip()
        if not detail:
            detail = f"probe exited with code {probe.returncode}"
        message = (
            f"FAISS GPU runtime probe failed for device_id={self._device_id}: {detail}. "
            "This typically means the installed FAISS GPU build does not support the visible GPU architecture."
        )
        self._runtime_probe_cache[cache_key] = message
        raise RuntimeError(message)
