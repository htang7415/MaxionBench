from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.datasets.d3_calibrate import calibrate_d3_params, paper_calibration_issues, write_d3_params_yaml
from maxionbench.datasets.d3_generator import default_d3_params, generate_synthetic_vectors


def test_calibrate_d3_produces_result_and_yaml(tmp_path: Path) -> None:
    vectors = generate_synthetic_vectors(num_vectors=1200, dim=32, seed=11)
    initial = default_d3_params(scale="10m", seed=11)
    result = calibrate_d3_params(vectors=vectors, initial_params=initial, seed=11, max_iters=2)
    assert result.iterations >= 1
    assert 0.0 <= result.eval.test_a_median_concentration <= 1.0

    out = tmp_path / "d3_params.yaml"
    write_d3_params_yaml(
        out,
        result.selected_params,
        eval_data=result.eval,
        calibration_metadata={"calibration_vector_count": 1200, "calibration_source": "synthetic_vectors"},
    )
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "k_clusters" in payload
    assert "beta_tenant" in payload
    assert "calibration_eval" in payload
    assert "recall_gap_50_minus_1_abs" in payload["calibration_eval"]
    assert "recall_gap_50_minus_1_negative" in payload["calibration_eval"]


def test_paper_calibration_issues_reports_negative_recall_gap_and_small_vector_count() -> None:
    payload = {
        "calibration_eval": {
            "test_a_median_concentration": 0.65,
            "test_b_cluster_spread": 20.0,
            "p99_ratio_1pct_to_50pct": 2.2,
            "recall_gap_50_minus_1": -0.10,
            "trivial": False,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    issues = paper_calibration_issues(payload=payload)
    assert any("negative" in msg for msg in issues)
    assert any("calibration_vector_count" in msg for msg in issues)
    assert any("synthetic/mock calibration" in msg for msg in issues)
