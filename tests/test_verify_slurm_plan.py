from __future__ import annotations

from pathlib import Path

from maxionbench.tools.verify_slurm_plan import verify_slurm_plan


def test_verify_slurm_plan_passes_for_repo_slurm_layout() -> None:
    summary = verify_slurm_plan(slurm_dir=Path("maxionbench/orchestration/slurm"), include_gpu=True)
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert int(summary["gpu_scenario_count"]) >= 3
    gpu_scenarios = [str(item) for item in summary["gpu_scenarios"]]
    assert any("track_b" in item for item in gpu_scenarios)
    assert any("track_c" in item for item in gpu_scenarios)


def test_verify_slurm_plan_detects_missing_d3_s1_baseline(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (slurm_dir / "conformance_matrix.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slurm_dir / "calibrate_d3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slurm_dir / "gpu_array.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slurm_dir / "cpu_array.sh").write_text(
        """#!/usr/bin/env bash
SCENARIOS=(
  "configs/scenarios/s1_ann_frontier.yaml"
  "configs/scenarios/s2_filtered_ann.yaml"
  "configs/scenarios/s3_churn_smooth.yaml"
  "configs/scenarios/s3b_churn_bursty.yaml"
  "configs/scenarios/s4_hybrid.yaml"
  "configs/scenarios/s6_fusion.yaml"
)
""",
        encoding="utf-8",
    )
    summary = verify_slurm_plan(slurm_dir=slurm_dir, include_gpu=True)
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("s1_ann_frontier_d3" in msg for msg in messages)


def test_verify_slurm_plan_skip_gpu_ignores_missing_gpu_array_script(tmp_path: Path) -> None:
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (slurm_dir / "conformance_matrix.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slurm_dir / "calibrate_d3.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slurm_dir / "cpu_array.sh").write_text(
        """#!/usr/bin/env bash
SCENARIOS=(
  "configs/scenarios/s1_ann_frontier.yaml"
  "configs/scenarios/s1_ann_frontier_d3.yaml"
  "configs/scenarios/s2_filtered_ann.yaml"
  "configs/scenarios/s3_churn_smooth.yaml"
  "configs/scenarios/s3b_churn_bursty.yaml"
  "configs/scenarios/s4_hybrid.yaml"
  "configs/scenarios/s6_fusion.yaml"
)
""",
        encoding="utf-8",
    )
    summary = verify_slurm_plan(slurm_dir=slurm_dir, include_gpu=False)
    assert summary["pass"] is True
