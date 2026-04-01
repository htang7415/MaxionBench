from __future__ import annotations

from pathlib import Path

from maxionbench.orchestration.local_preflight import evaluate_local_preflight


def test_local_preflight_uses_d2_manifest_sizes(tmp_path: Path) -> None:
    summary = evaluate_local_preflight(
        config_path=Path("configs/scenarios_paper/s1_ann_frontier_d2.yaml"),
        scratch_dir=tmp_path,
    )

    assert summary["dataset_bundle"] == "D2"
    assert summary["dataset_bytes"] == 21_474_836_480
    assert summary["engine_bytes"] == 42_949_672_960
    assert summary["temp_bytes"] == 8_589_934_592
    assert summary["required_bytes"] == 131_425_999_257
    assert summary["safety_factor"] == 1.8


def test_local_preflight_uses_d3_manifest_sizes(tmp_path: Path) -> None:
    summary = evaluate_local_preflight(
        config_path=Path("configs/scenarios_paper/s2_filtered_ann.yaml"),
        scratch_dir=tmp_path,
    )

    assert summary["dataset_bundle"] == "D3"
    assert summary["dataset_bytes"] == 32_212_254_720
    assert summary["engine_bytes"] == 51_539_607_552
    assert summary["temp_bytes"] == 12_884_901_888
    assert summary["required_bytes"] == 173_946_175_488
    assert summary["safety_factor"] == 1.8
