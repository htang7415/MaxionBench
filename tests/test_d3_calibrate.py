from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.datasets.d3_calibrate import calibrate_d3_params, write_d3_params_yaml
from maxionbench.datasets.d3_generator import default_d3_params, generate_synthetic_vectors


def test_calibrate_d3_produces_result_and_yaml(tmp_path: Path) -> None:
    vectors = generate_synthetic_vectors(num_vectors=1200, dim=32, seed=11)
    initial = default_d3_params(scale="10m", seed=11)
    result = calibrate_d3_params(vectors=vectors, initial_params=initial, seed=11, max_iters=2)
    assert result.iterations >= 1
    assert 0.0 <= result.eval.test_a_median_concentration <= 1.0

    out = tmp_path / "d3_params.yaml"
    write_d3_params_yaml(out, result.selected_params, eval_data=result.eval)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "k_clusters" in payload
    assert "beta_tenant" in payload
    assert "calibration_eval" in payload
