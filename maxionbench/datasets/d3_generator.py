"""D3 filtered-ANN dataset generator with correlated metadata.

Implements the pinned correlation model described in document.md Section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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
    ids: list[str]
    vectors: np.ndarray
    cluster_ids: np.ndarray
    tenant_ids: np.ndarray
    acl_buckets: np.ndarray
    time_buckets: np.ndarray
    payloads: list[dict[str, Any]]
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

    ids = [f"{id_prefix}-{index:07d}" for index in range(n_rows)]
    payloads: list[dict[str, Any]] = []
    for index in range(n_rows):
        payloads.append(
            {
                "tenant_id": f"tenant-{int(tenant_ids[index]):03d}",
                "tenant": f"tenant-{int(tenant_ids[index]):03d}",
                "acl_bucket": int(acl_buckets[index]),
                "time_bucket": int(time_buckets[index]),
                "cluster_id": int(cluster_ids[index]),
            }
        )
    return D3Dataset(
        ids=ids,
        vectors=vectors_np,
        cluster_ids=cluster_ids,
        tenant_ids=tenant_ids,
        acl_buckets=acl_buckets,
        time_buckets=time_buckets,
        payloads=payloads,
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
        filtered_idx = np.where(dataset.tenant_ids == tenant)[0]
        if filtered_idx.size == 0:
            continue
        scores = dataset.vectors[filtered_idx] @ dataset.vectors[q_idx]
        order = np.argsort(-scores, kind="stable")[:top_k]
        result_idx = filtered_idx[order]
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
    points_norm = np.sum(points * points, axis=1, keepdims=True)
    centroids_norm = np.sum(centroids * centroids, axis=1)[None, :]
    dists = points_norm + centroids_norm - 2.0 * (points @ centroids.T)
    return np.argmin(dists, axis=1)
