from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.tools.verify_d3_calibration import verify_d3_calibration_file


def test_verify_d3_calibration_file_flags_non_paper_ready_payload(tmp_path: Path) -> None:
    path = tmp_path / "d3_params_bad.yaml"
    payload = {
        "k_clusters": 4096,
        "calibration_eval": {
            "test_a_median_concentration": 0.54,
            "test_b_cluster_spread": 42.7,
            "p99_ratio_1pct_to_50pct": 1.19,
            "recall_gap_50_minus_1": -0.79,
            "trivial": True,
        },
        "calibration_vector_count": 10000,
        "calibration_source": "synthetic_vectors",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_d3_calibration_file(path=path)
    assert summary["paper_ready"] is False
    assert int(summary["error_count"]) > 0


def test_verify_d3_calibration_file_passes_for_paper_ready_payload(tmp_path: Path) -> None:
    path = tmp_path / "d3_params_ok.yaml"
    payload = {
        "k_clusters": 4096,
        "calibration_eval": {
            "test_a_median_concentration": 0.61,
            "test_b_cluster_spread": 24.0,
            "p99_ratio_1pct_to_50pct": 2.3,
            "recall_gap_50_minus_1": 0.08,
            "trivial": False,
        },
        "calibration_vector_count": 10000000,
        "calibration_source": "real_dataset_path",
        "calibration_paper_ready": True,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = verify_d3_calibration_file(path=path)
    assert summary["paper_ready"] is True
    assert int(summary["error_count"]) == 0
