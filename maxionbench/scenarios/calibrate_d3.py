"""Scenario helper for D3 calibration job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maxionbench.datasets.d3_calibrate import CalibrationResult, calibrate_d3_params, write_d3_params_yaml
from maxionbench.datasets.d3_generator import D3Params, generate_synthetic_vectors


@dataclass(frozen=True)
class CalibrateD3Config:
    vector_dim: int
    num_vectors: int
    seed: int
    output_params_path: str
    initial_params: D3Params


def run(cfg: CalibrateD3Config) -> CalibrationResult:
    vectors = generate_synthetic_vectors(num_vectors=cfg.num_vectors, dim=cfg.vector_dim, seed=cfg.seed)
    result = calibrate_d3_params(vectors=vectors, initial_params=cfg.initial_params, seed=cfg.seed)
    write_d3_params_yaml(Path(cfg.output_params_path), result.selected_params, eval_data=result.eval)
    return result
