"""D3 filtered-ANN dataset generator with correlated metadata.

Implements the pinned correlation model described in document.md Section 4.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_NEAREST_CENTROID_WORKING_SET_BYTES = 64 * 1024 * 1024
_TOPK_SCORE_CHUNK_ROWS = 65_536


class SequentialDocIdSequence(Sequence[str]):
    def __init__(self, size: int, *, prefix: str = "doc") -> None:
        self._size = int(size)
        self._prefix = str(prefix)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            return [self._format_id(position) for position in range(start, stop, step)]
        normalized = self._normalize_index(index)
        return self._format_id(normalized)

    def _normalize_index(self, index: int) -> int:
        value = int(index)
        if value < 0:
            value += self._size
        if value < 0 or value >= self._size:
            raise IndexError("doc id index out of range")
        return value

    def _format_id(self, position: int) -> str:
        return f"{self._prefix}-{position:07d}"


class GeneratedPayloadSequence(Sequence[dict[str, Any]]):
    def __init__(
        self,
        *,
        tenant_ids: np.ndarray,
        acl_buckets: np.ndarray,
        time_buckets: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> None:
        self._tenant_ids = tenant_ids
        self._acl_buckets = acl_buckets
        self._time_buckets = time_buckets
        self._cluster_ids = cluster_ids

    def __len__(self) -> int:
        return int(self._tenant_ids.shape[0])

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._payload_at(position) for position in range(start, stop, step)]
        normalized = self._normalize_index(index)
        return self._payload_at(normalized)

    def _normalize_index(self, index: int) -> int:
        value = int(index)
        size = len(self)
        if value < 0:
            value += size
        if value < 0 or value >= size:
            raise IndexError("payload index out of range")
        return value

    def _payload_at(self, position: int) -> dict[str, Any]:
        return {
            "tenant_id": f"tenant-{int(self._tenant_ids[position]):03d}",
            "tenant": f"tenant-{int(self._tenant_ids[position]):03d}",
            "acl_bucket": int(self._acl_buckets[position]),
            "time_bucket": int(self._time_buckets[position]),
            "cluster_id": int(self._cluster_ids[position]),
        }


@dataclass(frozen=True)
class D3Params:
    k_clusters: int
    num_tenants: int
    num_acl_buckets: int
    num_time_buckets: int
    beta_tenant: float
    beta_acl: float
    beta_time: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_clusters": self.k_clusters,
            "num_tenants": self.num_tenants,
            "num_acl_buckets": self.num_acl_buckets,
            "num_time_buckets": self.num_time_buckets,
            "beta_tenant": self.beta_tenant,
            "beta_acl": self.beta_acl,
            "beta_time": self.beta_time,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class D3Dataset:
    ids: Sequence[str]
    vectors: np.ndarray
    cluster_ids: np.ndarray
    tenant_ids: np.ndarray
    acl_buckets: np.ndarray
    time_buckets: np.ndarray
    payloads: Sequence[dict[str, Any]]
    params: D3Params


def default_d3_params(scale: str = "10m", seed: int = 42) -> D3Params:
    if scale == "50m":
        return D3Params(
            k_clusters=8192,
            num_tenants=100,
            num_acl_buckets=16,
            num_time_buckets=52,
            beta_tenant=0.75,
            beta_acl=0.70,
            beta_time=0.65,
            seed=seed,
        )
    return D3Params(
        k_clusters=4096,
        num_tenants=100,
        num_acl_buckets=16,
        num_time_buckets=52,
        beta_tenant=0.75,
        beta_acl=0.70,
        beta_time=0.65,
        seed=seed,
    )


def params_from_mapping(mapping: Mapping[str, Any], seed: int | None = None) -> D3Params:
    base_seed = int(mapping.get("seed", 42 if seed is None else seed))
    return D3Params(
        k_clusters=int(mapping["k_clusters"]),
        num_tenants=int(mapping["num_tenants"]),
        num_acl_buckets=int(mapping["num_acl_buckets"]),
        num_time_buckets=int(mapping["num_time_buckets"]),
        beta_tenant=float(mapping["beta_tenant"]),
        beta_acl=float(mapping["beta_acl"]),
        beta_time=float(mapping["beta_time"]),
        seed=base_seed,
    )


def generate_d3_dataset(
    vectors: np.ndarray,
    params: D3Params,
    *,
    id_prefix: str = "doc",
) -> D3Dataset:
    vectors_np = np.asarray(vectors, dtype=np.float32)
    if vectors_np.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    n_rows = int(vectors_np.shape[0])
    if n_rows < 2:
        raise ValueError("D3 generation requires at least 2 vectors")

    k = max(2, min(int(params.k_clusters), n_rows))
    cluster_ids = _cluster_vectors(vectors_np, k_clusters=k, seed=params.seed).astype(np.int32)
    rng = np.random.default_rng(params.seed)

    pref_tenant = rng.integers(0, params.num_tenants, size=k, endpoint=False)
    pref_acl = rng.integers(0, params.num_acl_buckets, size=k, endpoint=False)
    pref_time = rng.integers(0, params.num_time_buckets, size=k, endpoint=False)

    tenant_ids = _sample_affinity(
        preferred=pref_tenant[cluster_ids],
        cardinality=params.num_tenants,
        beta=params.beta_tenant,
        rng=rng,
    )
    acl_buckets = _sample_affinity(
        preferred=pref_acl[cluster_ids],
        cardinality=params.num_acl_buckets,
        beta=params.beta_acl,
        rng=rng,
    )
    time_buckets = _sample_affinity(
        preferred=pref_time[cluster_ids],
        cardinality=params.num_time_buckets,
        beta=params.beta_time,
        rng=rng,
    )

    return D3Dataset(
        ids=SequentialDocIdSequence(n_rows, prefix=id_prefix),
        vectors=vectors_np,
        cluster_ids=cluster_ids,
        tenant_ids=tenant_ids,
        acl_buckets=acl_buckets,
        time_buckets=time_buckets,
        payloads=GeneratedPayloadSequence(
            tenant_ids=tenant_ids,
            acl_buckets=acl_buckets,
            time_buckets=time_buckets,
            cluster_ids=cluster_ids,
        ),
        params=params,
    )


def generate_synthetic_vectors(
    *,
    num_vectors: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((num_vectors, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def tenant_top10_concentration(dataset: D3Dataset, top_n: int = 10) -> float:
    values: list[float] = []
    tenants = dataset.tenant_ids
    clusters = dataset.cluster_ids
    for tenant in range(dataset.params.num_tenants):
        mask = tenants == tenant
        total = int(np.count_nonzero(mask))
        if total == 0:
            continue
        tenant_clusters = clusters[mask]
        counts = np.bincount(tenant_clusters, minlength=int(clusters.max()) + 1)
        if counts.size == 0:
            continue
        probs = counts.astype(np.float64) / max(total, 1)
        probs = np.sort(probs)[::-1]
        values.append(float(np.sum(probs[:top_n])))
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def cluster_spread_at_one_percent(
    dataset: D3Dataset,
    *,
    num_queries: int = 200,
    top_k: int = 100,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    n = dataset.vectors.shape[0]
    query_count = min(num_queries, n)
    query_indices = rng.choice(n, size=query_count, replace=False)
    spreads: list[float] = []
    for q_idx in query_indices:
        tenant = int(dataset.tenant_ids[q_idx])
        filtered_mask = dataset.tenant_ids == tenant
        if not np.any(filtered_mask):
            continue
        result_idx = topk_masked_indices(
            dataset.vectors,
            dataset.vectors[q_idx],
            top_k=top_k,
            mask=filtered_mask,
        )
        unique_clusters = np.unique(dataset.cluster_ids[result_idx]).size
        spreads.append(float(unique_clusters))
    if not spreads:
        return float("inf")
    return float(np.mean(np.asarray(spreads, dtype=np.float64)))


def _sample_affinity(
    *,
    preferred: np.ndarray,
    cardinality: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if cardinality < 2:
        return np.zeros_like(preferred, dtype=np.int32)
    keep_pref = rng.random(preferred.shape[0]) < beta
    alternative = rng.integers(0, cardinality - 1, size=preferred.shape[0], endpoint=False)
    alternative = alternative + (alternative >= preferred).astype(np.int32)
    values = np.where(keep_pref, preferred, alternative)
    return values.astype(np.int32)


def _cluster_vectors(vectors: np.ndarray, *, k_clusters: int, seed: int) -> np.ndarray:
    """Lightweight NumPy mini-batch k-means to avoid binary sklearn runtime issues."""

    n, dim = vectors.shape
    rng = np.random.default_rng(seed)
    if k_clusters >= n:
        return np.arange(n, dtype=np.int32)

    init_idx = rng.choice(n, size=k_clusters, replace=False)
    centroids = vectors[init_idx].copy()
    counts = np.ones(k_clusters, dtype=np.float64)
    batch_size = min(4096, n)
    steps = max(10, min(50, n // max(1, batch_size // 2)))

    for _ in range(steps):
        batch_idx = rng.choice(n, size=batch_size, replace=False)
        batch = vectors[batch_idx]
        labels = _nearest_centroid(batch, centroids)
        for row, label in zip(batch, labels):
            counts[label] += 1.0
            eta = 1.0 / counts[label]
            centroids[label] = (1.0 - eta) * centroids[label] + eta * row

    return _nearest_centroid(vectors, centroids).astype(np.int32)


def _nearest_centroid(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_points == 0:
        return np.empty(0, dtype=np.int64)
    k_clusters = int(centroids.shape[0])
    bytes_per_score_row = max(1, k_clusters * np.dtype(np.float32).itemsize)
    batch_rows = max(1, min(n_points, _NEAREST_CENTROID_WORKING_SET_BYTES // bytes_per_score_row))
    labels = np.empty(n_points, dtype=np.int64)
    centroids_norm = np.sum(centroids * centroids, axis=1, dtype=np.float32)[None, :]
    for start in range(0, n_points, batch_rows):
        stop = min(n_points, start + batch_rows)
        chunk = np.asarray(points[start:stop], dtype=np.float32)
        scores = np.asarray(chunk @ centroids.T, dtype=np.float32)
        scores *= -2.0
        scores += centroids_norm
        scores += np.sum(chunk * chunk, axis=1, dtype=np.float32, keepdims=True)
        labels[start:stop] = np.argmin(scores, axis=1)
    return labels


def topk_masked_indices(
    vectors: np.ndarray,
    query_vec: np.ndarray,
    *,
    top_k: int,
    mask: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
    chunk_rows: int = _TOPK_SCORE_CHUNK_ROWS,
) -> np.ndarray:
    if top_k < 1:
        return np.empty(0, dtype=np.int64)
    if mask is not None and candidate_indices is not None:
        raise ValueError("mask and candidate_indices are mutually exclusive")
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    if query_vec.ndim != 1:
        raise ValueError("query_vec must be 1D")
    if vectors.shape[1] != query_vec.shape[0]:
        raise ValueError("query_vec dimension mismatch")

    best_scores = np.empty(0, dtype=np.float32)
    best_indices = np.empty(0, dtype=np.int64)
    if candidate_indices is not None:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        total = int(candidate_indices.shape[0])
        for start in range(0, total, max(1, int(chunk_rows))):
            stop = min(total, start + max(1, int(chunk_rows)))
            idx_chunk = candidate_indices[start:stop]
            if idx_chunk.size == 0:
                continue
            score_chunk = np.asarray(vectors[idx_chunk] @ query_vec, dtype=np.float32)
            best_scores, best_indices = _merge_topk_indices(
                best_scores=best_scores,
                best_indices=best_indices,
                new_scores=score_chunk,
                new_indices=idx_chunk,
                top_k=top_k,
            )
    else:
        total = int(vectors.shape[0])
        for start in range(0, total, max(1, int(chunk_rows))):
            stop = min(total, start + max(1, int(chunk_rows)))
            score_chunk = np.asarray(vectors[start:stop] @ query_vec, dtype=np.float32)
            if mask is None:
                idx_chunk = np.arange(start, stop, dtype=np.int64)
            else:
                mask_chunk = np.asarray(mask[start:stop], dtype=bool)
                if not np.any(mask_chunk):
                    continue
                local_indices = np.flatnonzero(mask_chunk).astype(np.int64, copy=False)
                idx_chunk = local_indices + start
                score_chunk = score_chunk[mask_chunk]
            best_scores, best_indices = _merge_topk_indices(
                best_scores=best_scores,
                best_indices=best_indices,
                new_scores=score_chunk,
                new_indices=idx_chunk,
                top_k=top_k,
            )
    if best_indices.size == 0:
        return best_indices
    order = np.lexsort((best_indices, -best_scores.astype(np.float64, copy=False)))
    return best_indices[order[:top_k]]


def _merge_topk_indices(
    *,
    best_scores: np.ndarray,
    best_indices: np.ndarray,
    new_scores: np.ndarray,
    new_indices: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if new_scores.size == 0:
        return best_scores, best_indices
    combined_scores = np.concatenate([best_scores, np.asarray(new_scores, dtype=np.float32)], axis=0)
    combined_indices = np.concatenate([best_indices, np.asarray(new_indices, dtype=np.int64)], axis=0)
    if combined_scores.size > top_k:
        keep = np.argpartition(-combined_scores, top_k - 1)[:top_k]
        combined_scores = combined_scores[keep]
        combined_indices = combined_indices[keep]
    order = np.lexsort((combined_indices, -combined_scores.astype(np.float64, copy=False)))
    return combined_scores[order], combined_indices[order]
