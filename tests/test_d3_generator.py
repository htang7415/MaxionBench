from __future__ import annotations

import numpy as np

from maxionbench.datasets.d3_generator import (
    D3Params,
    cluster_spread_at_one_percent,
    generate_d3_dataset,
    generate_synthetic_vectors,
    tenant_top10_concentration,
)


def test_generate_d3_dataset_shapes() -> None:
    vectors = generate_synthetic_vectors(num_vectors=300, dim=16, seed=7)
    params = D3Params(
        k_clusters=64,
        num_tenants=20,
        num_acl_buckets=8,
        num_time_buckets=12,
        beta_tenant=0.75,
        beta_acl=0.7,
        beta_time=0.65,
        seed=7,
    )
    d3 = generate_d3_dataset(vectors, params)
    assert d3.vectors.shape == (300, 16)
    assert len(d3.ids) == 300
    assert len(d3.payloads) == 300
    assert d3.cluster_ids.shape[0] == 300


def test_d3_metrics_return_finite_values() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((500, 24), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    params = D3Params(
        k_clusters=80,
        num_tenants=25,
        num_acl_buckets=10,
        num_time_buckets=16,
        beta_tenant=0.8,
        beta_acl=0.75,
        beta_time=0.7,
        seed=1,
    )
    d3 = generate_d3_dataset(vectors, params)
    test_a = tenant_top10_concentration(d3)
    test_b = cluster_spread_at_one_percent(d3, num_queries=30, top_k=40, seed=3)
    assert 0.0 <= test_a <= 1.0
    assert np.isfinite(test_b)
