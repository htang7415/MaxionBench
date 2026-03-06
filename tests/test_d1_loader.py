from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from maxionbench.datasets.loaders.d1_ann_hdf5 import load_d1_ann_hdf5


def test_load_d1_ann_hdf5_with_neighbors(tmp_path: Path) -> None:
    path = tmp_path / "d1.hdf5"
    train = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    test = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    neighbors = np.asarray([[0, 2], [1, 2]], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
        handle.create_dataset("neighbors", data=neighbors)
        handle.attrs["distance"] = "ip"

    data = load_d1_ann_hdf5(path, top_k=2)
    assert data.vectors.shape == (3, 2)
    assert data.queries.shape == (2, 2)
    assert data.ground_truth_ids[0][0] == "doc-0000000"
    assert len(data.ground_truth_ids) == 2


def test_load_d1_ann_hdf5_enforces_expected_sha256(tmp_path: Path) -> None:
    path = tmp_path / "d1.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=np.asarray([[1.0, 0.0]], dtype=np.float32))
        handle.create_dataset("test", data=np.asarray([[1.0, 0.0]], dtype=np.float32))
        handle.create_dataset("neighbors", data=np.asarray([[0]], dtype=np.int64))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    data = load_d1_ann_hdf5(path, top_k=1, expected_sha256=expected)
    assert data.vectors.shape == (1, 2)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_d1_ann_hdf5(path, top_k=1, expected_sha256=("0" * 64))
