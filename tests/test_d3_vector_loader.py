from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from maxionbench.datasets.loaders.d3_vectors import load_d3_vectors


def _unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def test_load_d3_vectors_supports_npy_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npy"
    np.save(path, _unit_vectors(20, 8, seed=3))
    loaded = load_d3_vectors(path, max_vectors=12, expected_dim=8)
    assert loaded.shape == (12, 8)
    assert loaded.dtype == np.float32


def test_load_d3_vectors_supports_npz_vectors_key(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    np.savez(path, vectors=_unit_vectors(15, 6, seed=5))
    loaded = load_d3_vectors(path, max_vectors=10, expected_dim=6)
    assert loaded.shape == (10, 6)


def test_load_d3_vectors_rejects_npz_without_vectors_key(tmp_path: Path) -> None:
    path = tmp_path / "vectors_bad.npz"
    np.savez(path, data=_unit_vectors(10, 4, seed=7))
    with pytest.raises(ValueError, match="must contain `vectors` array"):
        load_d3_vectors(path, max_vectors=5, expected_dim=4)


def test_load_d3_vectors_rejects_dimension_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "vectors_dim.npy"
    np.save(path, _unit_vectors(10, 5, seed=11))
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_d3_vectors(path, max_vectors=8, expected_dim=4)


def test_load_d3_vectors_rejects_short_dataset(tmp_path: Path) -> None:
    path = tmp_path / "vectors_short.npy"
    np.save(path, _unit_vectors(6, 4, seed=13))
    with pytest.raises(ValueError, match="fewer than requested"):
        load_d3_vectors(path, max_vectors=9, expected_dim=4)
