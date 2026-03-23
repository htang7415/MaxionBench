"""FAISS GPU adapter."""

from __future__ import annotations

from typing import Any

from .faiss_cpu import FaissCpuAdapter


class FaissGpuAdapter(FaissCpuAdapter):
    """FAISS GPU adapter based on FAISS CPU index transfer."""

    def __init__(self, device_id: int = 0) -> None:
        super().__init__()
        if not hasattr(self._faiss, "StandardGpuResources"):
            raise RuntimeError(
                "Installed FAISS does not include GPU support. Install a GPU-enabled FAISS wheel such as "
                "`faiss-gpu-cu12` for FaissGpuAdapter."
            )
        self._device_id = int(device_id)
        self._gpu_res = self._faiss.StandardGpuResources()

    def _finalize_index(self, cpu_index: Any) -> Any:
        return self._faiss.index_cpu_to_gpu(self._gpu_res, self._device_id, cpu_index)
