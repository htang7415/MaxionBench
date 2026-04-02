from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from maxionbench.datasets import d3_calibrate as d3_calibrate_mod
from maxionbench.datasets.d3_calibrate import CalibrationEval, calibrate_d3_params, paper_calibration_issues, write_d3_params_yaml
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
    assert "p99_ratio_50pct_to_1pct" in payload["calibration_eval"]
    assert "recall_gap_1_minus_50" in payload["calibration_eval"]
    assert "recall_gap_1_minus_50_abs" in payload["calibration_eval"]
    assert "recall_gap_50_minus_1_abs" in payload["calibration_eval"]
    assert "recall_gap_50_minus_1_negative" in payload["calibration_eval"]


def test_paper_calibration_issues_reports_small_gap_and_small_vector_count() -> None:
    payload = {
        "calibration_eval": {
            "test_a_median_concentration": 0.65,
            "test_b_cluster_spread": 20.0,
            "p99_ratio_50pct_to_1pct": 2.2,
            "recall_gap_1_minus_50": 0.01,
            "trivial": False,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    issues = paper_calibration_issues(payload=payload)
    assert any("recall_gap_1_minus_50" in msg for msg in issues)
    assert any("calibration_vector_count" in msg for msg in issues)
    assert any("synthetic/mock calibration" in msg for msg in issues)


def test_calibrate_d3_returns_last_evaluated_params_when_search_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = generate_synthetic_vectors(num_vectors=64, dim=8, seed=19)
    initial = default_d3_params(scale="10m", seed=19)
    seen_params = []

    def _fake_generate(vectors_arg, params):  # type: ignore[no-untyped-def]
        seen_params.append(params)
        return params

    def _fake_evaluate(dataset, *, seed, top_k):  # type: ignore[no-untyped-def]
        return CalibrationEval(
            test_a_median_concentration=0.10,
            test_b_cluster_spread=10.0,
            p99_1pct_ms=1.0,
            p99_50pct_ms=5.0,
            p99_ratio_1pct_to_50pct=0.2,
            p99_ratio_50pct_to_1pct=5.0,
            recall_1pct=0.8,
            recall_50pct=0.1,
            recall_gap_50_minus_1=-0.7,
            recall_gap_1_minus_50=0.7,
            trivial=True,
        )

    monkeypatch.setattr(d3_calibrate_mod, "generate_d3_dataset", _fake_generate)
    monkeypatch.setattr(d3_calibrate_mod, "evaluate_calibration", _fake_evaluate)

    result = calibrate_d3_params(vectors=vectors, initial_params=initial, seed=19, max_iters=2, beta_step=0.05, max_beta=1.0)

    assert len(seen_params) == 2
    assert result.selected_params == seen_params[-1]
    assert result.selected_params.beta_tenant == pytest.approx(initial.beta_tenant + 0.05)
