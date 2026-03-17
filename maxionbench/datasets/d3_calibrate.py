"""Calibration workflow for D3 metadata affinity parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import yaml

from maxionbench.datasets.d3_generator import (
    D3Dataset,
    D3Params,
    cluster_spread_at_one_percent,
    generate_d3_dataset,
    tenant_top10_concentration,
    topk_masked_indices,
)
from maxionbench.metrics.latency import percentile_ms
from maxionbench.metrics.quality import recall_at_k

MIN_TEST_A_MEDIAN_CONCENTRATION = 0.60
MAX_TEST_B_CLUSTER_SPREAD = 25.0
MIN_P99_RATIO_1PCT_TO_50PCT = 2.0
MIN_RECALL_GAP_50_MINUS_1 = 0.05
PAPER_MIN_CALIBRATION_VECTORS = 10_000_000


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


def is_trivial_curve(*, p99_ratio_1pct_to_50pct: float, recall_gap_50_minus_1: float) -> bool:
    """Documented trivial-curve criterion for D3 calibration.

    Trivial when either latency separation is too small OR recall separation is too small:
    p99_1% / p99_50% < 2.0 OR E_q[Recall@10_50% - Recall@10_1%] < 0.05
    """

    # Negative signed gap (recall@10_50% < recall@10_1%) still counts as trivial
    # under the pinned document criterion because it is strictly < 0.05.
    return (
        p99_ratio_1pct_to_50pct < MIN_P99_RATIO_1PCT_TO_50PCT
        or recall_gap_50_minus_1 < MIN_RECALL_GAP_50_MINUS_1
    )


def calibration_eval_passes_thresholds(eval_data: CalibrationEval) -> bool:
    return (
        eval_data.test_a_median_concentration >= MIN_TEST_A_MEDIAN_CONCENTRATION
        and eval_data.test_b_cluster_spread <= MAX_TEST_B_CLUSTER_SPREAD
        and not eval_data.trivial
    )


def paper_calibration_issues(
    *,
    payload: Mapping[str, Any],
    min_vectors: int = PAPER_MIN_CALIBRATION_VECTORS,
) -> list[str]:
    issues: list[str] = []
    eval_payload = payload.get("calibration_eval")
    if not isinstance(eval_payload, Mapping):
        issues.append("missing `calibration_eval` mapping")
        return issues

    test_a = _as_float(eval_payload.get("test_a_median_concentration"))
    if test_a is None:
        issues.append("calibration_eval.test_a_median_concentration must be numeric")
    elif test_a < MIN_TEST_A_MEDIAN_CONCENTRATION:
        issues.append(
            "calibration_eval.test_a_median_concentration must be >= "
            f"{MIN_TEST_A_MEDIAN_CONCENTRATION:.2f} (got {test_a:.6f})"
        )

    test_b = _as_float(eval_payload.get("test_b_cluster_spread"))
    if test_b is None:
        issues.append("calibration_eval.test_b_cluster_spread must be numeric")
    elif test_b > MAX_TEST_B_CLUSTER_SPREAD:
        issues.append(
            "calibration_eval.test_b_cluster_spread must be <= "
            f"{MAX_TEST_B_CLUSTER_SPREAD:.1f} (got {test_b:.6f})"
        )

    p99_ratio = _as_float(eval_payload.get("p99_ratio_1pct_to_50pct"))
    if p99_ratio is None:
        issues.append("calibration_eval.p99_ratio_1pct_to_50pct must be numeric")
    elif p99_ratio < MIN_P99_RATIO_1PCT_TO_50PCT:
        issues.append(
            "calibration_eval.p99_ratio_1pct_to_50pct must be >= "
            f"{MIN_P99_RATIO_1PCT_TO_50PCT:.1f} (got {p99_ratio:.6f})"
        )

    recall_gap = _as_float(eval_payload.get("recall_gap_50_minus_1"))
    if recall_gap is None:
        issues.append("calibration_eval.recall_gap_50_minus_1 must be numeric")
    else:
        if recall_gap < 0.0:
            issues.append(
                "calibration_eval.recall_gap_50_minus_1 is negative; "
                "this inverts Recall@10_50% vs Recall@10_1% ordering and is not paper-ready"
            )
        if recall_gap < MIN_RECALL_GAP_50_MINUS_1:
            issues.append(
                "calibration_eval.recall_gap_50_minus_1 must be >= "
                f"{MIN_RECALL_GAP_50_MINUS_1:.2f} (got {recall_gap:.6f})"
            )

    trivial = eval_payload.get("trivial")
    if not isinstance(trivial, bool):
        issues.append("calibration_eval.trivial must be a boolean")
    elif trivial:
        issues.append("calibration_eval.trivial must be false for paper-ready calibration")

    vector_count = _as_int(payload.get("calibration_vector_count"))
    if vector_count is None:
        issues.append("missing numeric `calibration_vector_count` in d3 params metadata")
    elif vector_count < int(min_vectors):
        issues.append(
            f"calibration_vector_count must be >= {int(min_vectors)} for paper-ready D3 calibration "
            f"(got {vector_count})"
        )

    source = str(payload.get("calibration_source") or "").strip().lower()
    if not source:
        issues.append("missing non-empty `calibration_source` in d3 params metadata")
    elif source.startswith("synthetic"):
        issues.append(
            f"calibration_source={source!r} indicates synthetic/mock calibration; "
            "paper runs require real LAION-subset calibration"
        )
    return issues


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
        passes_tests = calibration_eval_passes_thresholds(current_eval)
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
    trivial = is_trivial_curve(
        p99_ratio_1pct_to_50pct=p99_ratio,
        recall_gap_50_minus_1=recall_gap,
    )
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


def write_d3_params_yaml(
    path: Path,
    params: D3Params,
    eval_data: CalibrationEval | None = None,
    *,
    calibration_metadata: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = params.as_dict()
    if eval_data is not None:
        payload["calibration_eval"] = {
            "test_a_median_concentration": eval_data.test_a_median_concentration,
            "test_b_cluster_spread": eval_data.test_b_cluster_spread,
            "p99_ratio_1pct_to_50pct": eval_data.p99_ratio_1pct_to_50pct,
            "recall_gap_50_minus_1": eval_data.recall_gap_50_minus_1,
            "recall_gap_50_minus_1_abs": abs(eval_data.recall_gap_50_minus_1),
            "recall_gap_50_minus_1_negative": bool(eval_data.recall_gap_50_minus_1 < 0.0),
            "trivial": eval_data.trivial,
        }
    if calibration_metadata:
        for key, value in calibration_metadata.items():
            payload[str(key)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exact_topk_ids(dataset: D3Dataset, query_vec: np.ndarray, mask: np.ndarray, *, top_k: int) -> list[int]:
    top_indices = topk_masked_indices(dataset.vectors, query_vec, top_k=top_k, mask=mask)
    return [int(index) for index in top_indices.tolist()]


def _approx_topk_ids(
    dataset: D3Dataset,
    query_vec: np.ndarray,
    mask: np.ndarray,
    *,
    query_cluster: int,
    top_k: int,
) -> list[int]:
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
    top_indices = topk_masked_indices(
        dataset.vectors,
        query_vec,
        top_k=top_k,
        candidate_indices=np.asarray(candidate_idx, dtype=np.int64),
    )
    return [int(index) for index in top_indices.tolist()]
