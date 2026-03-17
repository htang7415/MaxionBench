"""Loader for D1 ann-benchmarks style HDF5 bundles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from maxionbench.datasets.cache_integrity import verify_file_sha256
from maxionbench.datasets.d3_generator import SequentialDocIdSequence


_TOPK_SCORE_CHUNK_ROWS = 65_536


@dataclass(frozen=True)
class D1AnnDataset:
    ids: Sequence[str]
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
    expected_sha256: str | None = None,
) -> D1AnnDataset:
    if not path.exists():
        raise FileNotFoundError(f"D1 HDF5 file not found: {path}")
    if expected_sha256 is not None:
        verify_file_sha256(path=path, expected_sha256=expected_sha256, label="D1 dataset_path")
    with h5py.File(path, "r") as handle:
        vectors = _read_first(handle, ["train", "vectors", "embeddings"])
        queries = _read_first(handle, ["test", "queries"])
        metric = str(handle.attrs.get("distance", "ip")).lower()
        neighbors = handle.get("neighbors")

        vectors_slice = vectors[:max_vectors] if max_vectors is not None else vectors
        queries_slice = queries[:max_queries] if max_queries is not None else queries
        vectors_np = np.asarray(vectors_slice).astype(np.float32, copy=False)
        queries_np = np.asarray(queries_slice).astype(np.float32, copy=False)

        ids = SequentialDocIdSequence(int(vectors_np.shape[0]))

        if neighbors is not None:
            gt_indices = np.asarray(neighbors[: queries_np.shape[0], :top_k], dtype=np.int64)
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
    ids: Sequence[str],
    top_k: int,
    metric: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    vector_norms = None
    normalized = metric.lower()
    if normalized in ("angular", "cos", "cosine"):
        vector_norms = np.linalg.norm(vectors, axis=1).astype(np.float32, copy=False)
    for query in queries:
        top_indices = _topk_indices_for_query(
            vectors=vectors,
            query=np.asarray(query, dtype=np.float32),
            top_k=top_k,
            metric=normalized,
            vector_norms=vector_norms,
        )
        rows.append([ids[int(index)] for index in top_indices])
    return rows


def _topk_indices_for_query(
    *,
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int,
    metric: str,
    vector_norms: np.ndarray | None = None,
) -> np.ndarray:
    best_scores = np.empty(0, dtype=np.float32)
    best_indices = np.empty(0, dtype=np.int64)
    query_norm = float(np.linalg.norm(query)) + 1e-12
    for start in range(0, int(vectors.shape[0]), _TOPK_SCORE_CHUNK_ROWS):
        stop = min(int(vectors.shape[0]), start + _TOPK_SCORE_CHUNK_ROWS)
        chunk = np.asarray(vectors[start:stop], dtype=np.float32)
        if metric in ("angular", "cos", "cosine"):
            assert vector_norms is not None
            score_chunk = np.asarray(chunk @ query, dtype=np.float32)
            score_chunk /= np.asarray(vector_norms[start:stop], dtype=np.float32) * query_norm
        elif metric in ("l2", "euclidean", "euclid"):
            diff = np.asarray(chunk - query, dtype=np.float32)
            score_chunk = -np.sum(diff * diff, axis=1, dtype=np.float32)
        else:
            score_chunk = np.asarray(chunk @ query, dtype=np.float32)
        best_scores, best_indices = _merge_topk_scores(
            best_scores=best_scores,
            best_indices=best_indices,
            new_scores=score_chunk,
            new_indices=np.arange(start, stop, dtype=np.int64),
            top_k=top_k,
        )
    return best_indices


def _merge_topk_scores(
    *,
    best_scores: np.ndarray,
    best_indices: np.ndarray,
    new_scores: np.ndarray,
    new_indices: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if new_scores.size == 0:
        return best_scores, best_indices
    if new_scores.size > top_k:
        keep = np.argpartition(-new_scores, top_k - 1)[:top_k]
        new_scores = new_scores[keep]
        new_indices = new_indices[keep]
    combined_scores = np.concatenate([best_scores, np.asarray(new_scores, dtype=np.float32)], axis=0)
    combined_indices = np.concatenate([best_indices, np.asarray(new_indices, dtype=np.int64)], axis=0)
    if combined_scores.size > top_k:
        keep = np.argpartition(-combined_scores, top_k - 1)[:top_k]
        combined_scores = combined_scores[keep]
        combined_indices = combined_indices[keep]
    order = np.lexsort((combined_indices, -combined_scores.astype(np.float64, copy=False)))
    return combined_scores[order], combined_indices[order]
