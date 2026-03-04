"""Loader for D1 ann-benchmarks style HDF5 bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class D1AnnDataset:
    ids: list[str]
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: list[list[str]]
    metric: str


def load_d1_ann_hdf5(
    path: Path,
    *,
    max_vectors: int | None = None,
    max_queries: int | None = None,
    top_k: int = 10,
) -> D1AnnDataset:
    if not path.exists():
        raise FileNotFoundError(f"D1 HDF5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        vectors = _read_first(handle, ["train", "vectors", "embeddings"])
        queries = _read_first(handle, ["test", "queries"])
        metric = str(handle.attrs.get("distance", "ip")).lower()
        neighbors = handle.get("neighbors")

        vectors_np = np.asarray(vectors, dtype=np.float32)
        queries_np = np.asarray(queries, dtype=np.float32)
        if max_vectors is not None:
            vectors_np = vectors_np[:max_vectors]
        if max_queries is not None:
            queries_np = queries_np[:max_queries]

        ids = [f"doc-{idx:07d}" for idx in range(vectors_np.shape[0])]

        if neighbors is not None:
            gt_indices = np.asarray(neighbors, dtype=np.int64)
            gt_indices = gt_indices[: queries_np.shape[0], :top_k]
            gt = [[ids[int(idx)] for idx in row if int(idx) < len(ids)] for row in gt_indices]
        else:
            gt = _exact_ground_truth(vectors_np, queries_np, ids=ids, top_k=top_k, metric=metric)
    return D1AnnDataset(
        ids=ids,
        vectors=vectors_np,
        queries=queries_np,
        ground_truth_ids=gt,
        metric=metric,
    )


def _read_first(handle: h5py.File, keys: Iterable[str]):
    for key in keys:
        if key in handle:
            return handle[key]
    available = sorted(handle.keys())
    raise KeyError(f"Expected one of keys {list(keys)} but found {available}")


def _exact_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    *,
    ids: list[str],
    top_k: int,
    metric: str,
) -> list[list[str]]:
    scores = _score_matrix(vectors=vectors, queries=queries, metric=metric)
    top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return [[ids[int(index)] for index in row] for row in top_indices]


def _score_matrix(vectors: np.ndarray, queries: np.ndarray, metric: str) -> np.ndarray:
    normalized = metric.lower()
    if normalized in ("angular", "cos", "cosine"):
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        return queries_norm @ vectors_norm.T
    if normalized in ("l2", "euclidean", "euclid"):
        diff = queries[:, None, :] - vectors[None, :, :]
        return -np.linalg.norm(diff, axis=2)
    return queries @ vectors.T
