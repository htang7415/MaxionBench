"""Calibration workflow for D3 metadata affinity parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from maxionbench.datasets.d3_generator import (
    D3Dataset,
    D3Params,
    cluster_spread_at_one_percent,
    generate_d3_dataset,
    tenant_top10_concentration,
)
from maxionbench.metrics.latency import percentile_ms
from maxionbench.metrics.quality import recall_at_k


@dataclass(frozen=True)
class CalibrationEval:
    test_a_median_concentration: float
    test_b_cluster_spread: float
    p99_1pct_ms: float
    p99_50pct_ms: float
    p99_ratio_1pct_to_50pct: float
    recall_1pct: float
    recall_50pct: float
    recall_gap_50_minus_1: float
    trivial: bool


@dataclass(frozen=True)
class CalibrationResult:
    selected_params: D3Params
    eval: CalibrationEval
    iterations: int
    adjusted: bool


def calibrate_d3_params(
    vectors: np.ndarray,
    initial_params: D3Params,
    *,
    seed: int = 42,
    max_iters: int = 5,
    beta_step: float = 0.05,
    top_k: int = 10,
) -> CalibrationResult:
    params = initial_params
    adjusted = False
    last_eval: CalibrationEval | None = None
    for iteration in range(1, max_iters + 1):
        dataset = generate_d3_dataset(vectors, params)
        current_eval = evaluate_calibration(dataset, seed=seed, top_k=top_k)
        last_eval = current_eval
        passes_tests = (
            current_eval.test_a_median_concentration >= 0.60
            and current_eval.test_b_cluster_spread <= 25.0
            and not current_eval.trivial
        )
        if passes_tests:
            return CalibrationResult(
                selected_params=params,
                eval=current_eval,
                iterations=iteration,
                adjusted=adjusted,
            )
        adjusted = True
        params = D3Params(
            k_clusters=params.k_clusters,
            num_tenants=params.num_tenants,
            num_acl_buckets=params.num_acl_buckets,
            num_time_buckets=params.num_time_buckets,
            beta_tenant=min(0.95, params.beta_tenant + beta_step),
            beta_acl=min(0.95, params.beta_acl + beta_step),
            beta_time=min(0.95, params.beta_time + beta_step),
            seed=params.seed,
        )
    assert last_eval is not None
    return CalibrationResult(
        selected_params=params,
        eval=last_eval,
        iterations=max_iters,
        adjusted=adjusted,
    )


def evaluate_calibration(
    dataset: D3Dataset,
    *,
    seed: int = 42,
    top_k: int = 10,
    num_queries: int = 120,
) -> CalibrationEval:
    test_a = tenant_top10_concentration(dataset, top_n=10)
    test_b = cluster_spread_at_one_percent(dataset, num_queries=num_queries, top_k=100, seed=seed)

    rng = np.random.default_rng(seed)
    n = dataset.vectors.shape[0]
    q_count = min(num_queries, n)
    query_ids = rng.choice(n, size=q_count, replace=False)

    lat_1pct: list[float] = []
    lat_50pct: list[float] = []
    rec_1pct: list[float] = []
    rec_50pct: list[float] = []

    for idx in query_ids:
        qvec = dataset.vectors[idx]
        tenant = dataset.tenant_ids[idx]
        acl_half = dataset.acl_buckets[idx] < (dataset.params.num_acl_buckets // 2)

        mask_1 = dataset.tenant_ids == tenant
        mask_50 = dataset.acl_buckets < (dataset.params.num_acl_buckets // 2) if acl_half else dataset.acl_buckets >= (
            dataset.params.num_acl_buckets // 2
        )

        gt_1 = _exact_topk_ids(dataset, qvec, mask_1, top_k=top_k)
        gt_50 = _exact_topk_ids(dataset, qvec, mask_50, top_k=top_k)

        t0 = time.perf_counter()
        apx_1 = _approx_topk_ids(dataset, qvec, mask_1, query_cluster=int(dataset.cluster_ids[idx]), top_k=top_k)
        lat_1pct.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        apx_50 = _approx_topk_ids(dataset, qvec, mask_50, query_cluster=int(dataset.cluster_ids[idx]), top_k=top_k)
        lat_50pct.append((time.perf_counter() - t1) * 1000.0)

        rec_1pct.append(recall_at_k(apx_1, gt_1, k=top_k))
        rec_50pct.append(recall_at_k(apx_50, gt_50, k=top_k))

    p99_1 = percentile_ms(lat_1pct, 99)
    p99_50 = percentile_ms(lat_50pct, 99)
    p99_ratio = 0.0 if p99_50 <= 0.0 else p99_1 / p99_50
    recall_1 = float(np.mean(np.asarray(rec_1pct, dtype=np.float64))) if rec_1pct else 0.0
    recall_50 = float(np.mean(np.asarray(rec_50pct, dtype=np.float64))) if rec_50pct else 0.0
    recall_gap = recall_50 - recall_1
    trivial = p99_ratio < 2.0 or recall_gap < 0.05
    return CalibrationEval(
        test_a_median_concentration=test_a,
        test_b_cluster_spread=test_b,
        p99_1pct_ms=p99_1,
        p99_50pct_ms=p99_50,
        p99_ratio_1pct_to_50pct=p99_ratio,
        recall_1pct=recall_1,
        recall_50pct=recall_50,
        recall_gap_50_minus_1=recall_gap,
        trivial=trivial,
    )


def write_d3_params_yaml(path: Path, params: D3Params, eval_data: CalibrationEval | None = None) -> None:
    payload: dict[str, Any] = params.as_dict()
    if eval_data is not None:
        payload["calibration_eval"] = {
            "test_a_median_concentration": eval_data.test_a_median_concentration,
            "test_b_cluster_spread": eval_data.test_b_cluster_spread,
            "p99_ratio_1pct_to_50pct": eval_data.p99_ratio_1pct_to_50pct,
            "recall_gap_50_minus_1": eval_data.recall_gap_50_minus_1,
            "trivial": eval_data.trivial,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def _exact_topk_ids(dataset: D3Dataset, query_vec: np.ndarray, mask: np.ndarray, *, top_k: int) -> list[str]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    scores = dataset.vectors[idx] @ query_vec
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [dataset.ids[int(idx[o])] for o in order]


def _approx_topk_ids(
    dataset: D3Dataset,
    query_vec: np.ndarray,
    mask: np.ndarray,
    *,
    query_cluster: int,
    top_k: int,
) -> list[str]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    same_cluster = idx[dataset.cluster_ids[idx] == query_cluster]
    other = idx[dataset.cluster_ids[idx] != query_cluster]
    if same_cluster.size == 0:
        candidate_idx = idx
    else:
        budget = min(max(top_k * 20, same_cluster.size), idx.size)
        fill = max(0, budget - same_cluster.size)
        if fill > 0 and other.size > 0:
            sampled_other = other[:fill]
            candidate_idx = np.concatenate([same_cluster, sampled_other], axis=0)
        else:
            candidate_idx = same_cluster
    scores = dataset.vectors[candidate_idx] @ query_vec
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [dataset.ids[int(candidate_idx[o])] for o in order]
