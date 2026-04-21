from __future__ import annotations

from pathlib import Path

from maxionbench.orchestration.local_preflight import evaluate_local_preflight


def test_local_preflight_uses_d4_manifest_sizes(tmp_path: Path) -> None:
    summary = evaluate_local_preflight(
        config_path=Path("configs/scenarios_portable/s1_single_hop.yaml"),
        scratch_dir=tmp_path,
    )

    assert summary["dataset_bundle"] == "D4"
    assert summary["dataset_bytes"] == 6_442_450_944
    assert summary["engine_bytes"] == 12_884_901_888
    assert summary["temp_bytes"] == 3_221_225_472
    assert summary["required_bytes"] == 40_587_440_947
    assert summary["safety_factor"] == 1.8


def test_local_preflight_falls_back_for_frames_portable(tmp_path: Path) -> None:
    summary = evaluate_local_preflight(
        config_path=Path("configs/scenarios_portable/s3_multi_hop.yaml"),
        scratch_dir=tmp_path,
    )

    assert summary["dataset_bundle"] == "FRAMES_PORTABLE"
    assert summary["dataset_bytes"] == 2304
    assert summary["engine_bytes"] == 2764
    assert summary["temp_bytes"] == 1382
    assert summary["required_bytes"] == 11610
    assert summary["safety_factor"] == 1.8
