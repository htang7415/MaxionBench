from __future__ import annotations

from pathlib import Path

import yaml


def test_paper_scenario_config_catalog_contains_expected_files() -> None:
    root = Path("configs/scenarios_paper")
    assert root.exists()
    expected = {
        "calibrate_d3.yaml",
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


def test_paper_scenario_configs_enforce_d3_10m_scale() -> None:
    root = Path("configs/scenarios_paper")
    for name in sorted(path.name for path in root.glob("*.yaml")):
        payload = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        bundle = str(payload.get("dataset_bundle", "")).upper()
        if bundle == "D3":
            assert int(payload.get("num_vectors", 0)) >= 10_000_000
            continue
        assert bundle == "D4"
        assert bool(payload.get("d4_use_real_data", False)) is True
