from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from maxionbench.datasets.loaders.d2_bigann import load_d2_bigann, read_fvecs, read_ivecs


def _write_fvecs(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    dim = arr.shape[1]
    packed = np.empty((arr.shape[0], dim + 1), dtype=np.int32)
    packed[:, 0] = dim
    packed[:, 1:] = arr.view(np.int32)
    packed.tofile(path)


def _write_ivecs(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int32)
    dim = arr.shape[1]
    packed = np.empty((arr.shape[0], dim + 1), dtype=np.int32)
    packed[:, 0] = dim
    packed[:, 1:] = arr
    packed.tofile(path)


def test_read_fvecs_and_ivecs_roundtrip(tmp_path: Path) -> None:
    vecs = np.array([[1.0, 2.0, 3.5], [4.0, -1.0, 0.5]], dtype=np.float32)
    ivecs = np.array([[1, 2, 3], [0, 2, 1]], dtype=np.int32)

    fpath = tmp_path / "x.fvecs"
    ipath = tmp_path / "y.ivecs"
    _write_fvecs(fpath, vecs)
    _write_ivecs(ipath, ivecs)

    got_f = read_fvecs(fpath)
    got_i = read_ivecs(ipath)
    np.testing.assert_allclose(got_f, vecs)
    np.testing.assert_array_equal(got_i, ivecs)


def test_load_d2_bigann_uses_provided_ground_truth(tmp_path: Path) -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0], [4.9, 5.1]], dtype=np.float32)
    gt = np.array([[0, 1], [2, 1]], dtype=np.int32)

    base_path = tmp_path / "base.fvecs"
    query_path = tmp_path / "query.fvecs"
    gt_path = tmp_path / "gt.ivecs"
    _write_fvecs(base_path, base)
    _write_fvecs(query_path, queries)
    _write_ivecs(gt_path, gt)

    ds = load_d2_bigann(base_fvecs=base_path, query_fvecs=query_path, gt_ivecs=gt_path, top_k=2)
    assert ds.ids == ["doc-0000000", "doc-0000001", "doc-0000002"]
    assert ds.ground_truth_ids[0] == ["doc-0000000", "doc-0000001"]
    assert ds.ground_truth_ids[1] == ["doc-0000002", "doc-0000001"]


def test_load_d2_bigann_enforces_expected_sha256(tmp_path: Path) -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)
    base_path = tmp_path / "base.fvecs"
    query_path = tmp_path / "query.fvecs"
    _write_fvecs(base_path, base)
    _write_fvecs(query_path, queries)

    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    query_sha = hashlib.sha256(query_path.read_bytes()).hexdigest()
    ds = load_d2_bigann(
        base_fvecs=base_path,
        query_fvecs=query_path,
        top_k=1,
        base_expected_sha256=base_sha,
        query_expected_sha256=query_sha,
    )
    assert ds.vectors.shape == (2, 2)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_d2_bigann(
            base_fvecs=base_path,
            query_fvecs=query_path,
            top_k=1,
            base_expected_sha256=("0" * 64),
        )
