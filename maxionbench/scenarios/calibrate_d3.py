"""Scenario helper for D3 calibration job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maxionbench.datasets.d3_calibrate import (
    PAPER_MIN_CALIBRATION_VECTORS,
    CalibrationResult,
    calibrate_d3_params,
    paper_calibration_issues,
    write_d3_params_yaml,
)
from maxionbench.datasets.d3_generator import D3Params, generate_synthetic_vectors


@dataclass(frozen=True)
class CalibrateD3Config:
    vector_dim: int
    num_vectors: int
    seed: int
    output_params_path: str
    initial_params: D3Params
    dataset_path: str | None = None
    calibration_source: str = "synthetic_vectors"
    calibration_dataset_hash: str = "synthetic-d3-calibration"


def run(cfg: CalibrateD3Config) -> CalibrationResult:
    vectors, source = _load_vectors_for_calibration(cfg)
    result = calibrate_d3_params(vectors=vectors, initial_params=cfg.initial_params, seed=cfg.seed)
    base_payload = {
        **result.selected_params.as_dict(),
        "calibration_eval": {
            "test_a_median_concentration": result.eval.test_a_median_concentration,
            "test_b_cluster_spread": result.eval.test_b_cluster_spread,
            "p99_ratio_1pct_to_50pct": result.eval.p99_ratio_1pct_to_50pct,
            "recall_gap_50_minus_1": result.eval.recall_gap_50_minus_1,
            "trivial": result.eval.trivial,
        },
        "calibration_vector_count": int(cfg.num_vectors),
        "calibration_source": str(source),
    }
    issues = paper_calibration_issues(payload=base_payload, min_vectors=PAPER_MIN_CALIBRATION_VECTORS)
    write_d3_params_yaml(
        Path(cfg.output_params_path),
        result.selected_params,
        eval_data=result.eval,
        calibration_metadata={
            "calibration_vector_count": int(cfg.num_vectors),
            "calibration_vector_dim": int(cfg.vector_dim),
            "calibration_source": str(source),
            "calibration_dataset_hash": str(cfg.calibration_dataset_hash),
            "calibration_paper_ready": len(issues) == 0,
            "calibration_paper_readiness_issues": issues,
        },
    )
    return result


def _load_vectors_for_calibration(cfg: CalibrateD3Config) -> tuple[np.ndarray, str]:
    if not cfg.dataset_path:
        return (
            generate_synthetic_vectors(num_vectors=cfg.num_vectors, dim=cfg.vector_dim, seed=cfg.seed),
            str(cfg.calibration_source),
        )
    dataset_path = Path(cfg.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"calibration dataset_path does not exist: {dataset_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(dataset_path, mmap_mode="r")
    elif suffix == ".npz":
        npz = np.load(dataset_path, allow_pickle=False)
        if "vectors" not in npz:
            raise ValueError("npz calibration dataset must contain `vectors` array")
        data = npz["vectors"]
    else:
        raise ValueError(
            f"unsupported calibration dataset format for {dataset_path}; supported extensions: .npy, .npz"
        )
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"calibration dataset must be 2D [N, D]; got shape={tuple(data.shape)}")
    if int(data.shape[0]) < int(cfg.num_vectors):
        raise ValueError(
            f"calibration dataset has {int(data.shape[0])} vectors, fewer than requested num_vectors={cfg.num_vectors}"
        )
    if int(data.shape[1]) != int(cfg.vector_dim):
        raise ValueError(
            f"calibration dataset dimension mismatch: expected {cfg.vector_dim}, got {int(data.shape[1])}"
        )
    return np.asarray(data[: cfg.num_vectors], dtype=np.float32), "real_dataset_path"
