"""Loader for D3 vector bundles used by filtered/churn scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maxionbench.datasets.cache_integrity import verify_file_sha256


def load_d3_vectors(
    path: Path,
    *,
    max_vectors: int,
    expected_dim: int,
    expected_sha256: str | None = None,
) -> np.ndarray:
    if max_vectors < 1:
        raise ValueError("max_vectors must be >= 1")
    if expected_dim < 1:
        raise ValueError("expected_dim must be >= 1")
    if not path.exists():
        raise FileNotFoundError(f"D3 vector file not found: {path}")
    if expected_sha256 is not None:
        verify_file_sha256(path=path, expected_sha256=expected_sha256, label="D3 dataset_path")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, mmap_mode="r")
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        if "vectors" not in npz:
            raise ValueError("D3 npz dataset must contain `vectors` array")
        data = npz["vectors"]
    else:
        raise ValueError(f"unsupported D3 dataset format for {path}; supported extensions: .npy, .npz")

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"D3 dataset must be 2D [N, D]; got shape={tuple(array.shape)}")
    if int(array.shape[0]) < int(max_vectors):
        raise ValueError(
            f"D3 dataset has {int(array.shape[0])} vectors, fewer than requested num_vectors={int(max_vectors)}"
        )
    if int(array.shape[1]) != int(expected_dim):
        raise ValueError(
            f"D3 dataset dimension mismatch: expected {int(expected_dim)}, got {int(array.shape[1])}"
        )
    view = array[:max_vectors]
    return np.asarray(view).astype(np.float32, copy=False)
