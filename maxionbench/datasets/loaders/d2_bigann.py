"""Loader for BigANN style .fvecs/.ivecs bundles (D2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maxionbench.datasets.cache_integrity import verify_file_sha256


@dataclass(frozen=True)
class D2BigAnnDataset:
    ids: list[str]
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: list[list[str]]
    metric: str = "l2"


def load_d2_bigann(
    *,
    base_fvecs: Path,
    query_fvecs: Path,
    gt_ivecs: Path | None = None,
    max_vectors: int | None = None,
    max_queries: int | None = None,
    top_k: int = 10,
    base_expected_sha256: str | None = None,
    query_expected_sha256: str | None = None,
    gt_expected_sha256: str | None = None,
) -> D2BigAnnDataset:
    if base_expected_sha256 is not None:
        verify_file_sha256(path=base_fvecs, expected_sha256=base_expected_sha256, label="D2 d2_base_fvecs_path")
    if query_expected_sha256 is not None:
        verify_file_sha256(path=query_fvecs, expected_sha256=query_expected_sha256, label="D2 d2_query_fvecs_path")
    if gt_ivecs is not None and gt_expected_sha256 is not None:
        verify_file_sha256(path=gt_ivecs, expected_sha256=gt_expected_sha256, label="D2 d2_gt_ivecs_path")

    vectors = read_fvecs(base_fvecs)
    queries = read_fvecs(query_fvecs)
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    if max_queries is not None:
        queries = queries[:max_queries]

    ids = [f"doc-{idx:07d}" for idx in range(vectors.shape[0])]
    if gt_ivecs is not None and gt_ivecs.exists():
        gt_idx = read_ivecs(gt_ivecs)
        gt_idx = gt_idx[: queries.shape[0], :top_k]
        gt = [[ids[int(i)] for i in row if int(i) < len(ids)] for row in gt_idx]
    else:
        gt = _exact_l2_ground_truth(vectors=vectors, queries=queries, ids=ids, top_k=top_k)
    return D2BigAnnDataset(
        ids=ids,
        vectors=vectors.astype(np.float32, copy=False),
        queries=queries.astype(np.float32, copy=False),
        ground_truth_ids=gt,
        metric="l2",
    )


def read_fvecs(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension in fvecs: {dim}")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"Corrupt fvecs file {path}: size {raw.size} not divisible by stride {stride}")
    rows = raw.reshape(-1, stride)
    if np.any(rows[:, 0] != dim):
        raise ValueError(f"Inconsistent vector dimensions in fvecs file {path}")
    vec_u32 = rows[:, 1:].astype(np.uint32, copy=False)
    return vec_u32.view(np.float32).copy()


def read_ivecs(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension in ivecs: {dim}")
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"Corrupt ivecs file {path}: size {raw.size} not divisible by stride {stride}")
    rows = raw.reshape(-1, stride)
    if np.any(rows[:, 0] != dim):
        raise ValueError(f"Inconsistent vector dimensions in ivecs file {path}")
    return rows[:, 1:].astype(np.int32, copy=True)


def _exact_l2_ground_truth(*, vectors: np.ndarray, queries: np.ndarray, ids: list[str], top_k: int) -> list[list[str]]:
    if vectors.size == 0 or queries.size == 0:
        return [[] for _ in range(queries.shape[0])]
    diff = queries[:, None, :] - vectors[None, :, :]
    scores = -np.linalg.norm(diff, axis=2)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return [[ids[int(idx)] for idx in row] for row in order]
