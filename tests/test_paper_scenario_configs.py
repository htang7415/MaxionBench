from __future__ import annotations

from pathlib import Path

import yaml


def test_paper_scenario_config_catalog_contains_expected_files() -> None:
    root = Path("configs/scenarios_paper")
    assert root.exists()
    expected = {
        "calibrate_d3.yaml",
        "s1_ann_frontier_d1_glove.yaml",
        "s1_ann_frontier_d1_sift.yaml",
        "s1_ann_frontier_d1_gist.yaml",
        "s1_ann_frontier_d2.yaml",
        "s1_ann_frontier_d3.yaml",
        "s2_filtered_ann.yaml",
        "s3_churn_smooth.yaml",
        "s3b_churn_bursty.yaml",
        "s4_hybrid.yaml",
        "s5_rerank.yaml",
        "s6_fusion.yaml",
    }
    actual = {path.name for path in root.glob("*.yaml")}
    assert actual == expected


def test_paper_scenario_configs_match_dataset_contracts() -> None:
    root = Path("configs/scenarios_paper")
    d1_expected = {
        "s1_ann_frontier_d1_glove.yaml": (
            "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D1/glove-100-angular",
            100,
            5000,
            200,
        ),
        "s1_ann_frontier_d1_sift.yaml": (
            "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D1/sift-128-euclidean",
            128,
            5000,
            200,
        ),
        "s1_ann_frontier_d1_gist.yaml": (
            "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D1/gist-960-euclidean",
            960,
            5000,
            200,
        ),
    }
    for name in sorted(path.name for path in root.glob("*.yaml")):
        payload = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        bundle = str(payload.get("dataset_bundle", "")).upper()
        if bundle == "D1":
            expected_path, expected_dim, expected_vectors, expected_queries = d1_expected[name]
            assert payload.get("processed_dataset_path") == expected_path
            assert int(payload.get("vector_dim", 0)) == expected_dim
            assert int(payload.get("num_vectors", 0)) == expected_vectors
            assert int(payload.get("num_queries", 0)) == expected_queries
            continue
        if bundle == "D2":
            assert payload.get("processed_dataset_path") == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D2/deep-image-96-angular"
            assert int(payload.get("vector_dim", 0)) == 96
            assert int(payload.get("num_vectors", 0)) == 10_000_000
            assert int(payload.get("num_queries", 0)) == 10_000
            continue
        if bundle == "D3":
            assert payload.get("dataset_path") == "${MAXIONBENCH_D3_DATASET_PATH}"
            assert int(payload.get("vector_dim", 0)) == 192
            assert int(payload.get("num_vectors", 0)) >= 10_000_000
            continue
        assert bundle == "D4"
        assert bool(payload.get("d4_use_real_data", False)) is True
