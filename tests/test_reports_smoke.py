from __future__ import annotations

from pathlib import Path
import json

import matplotlib.image as mpimg
import pandas as pd
import pytest
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.runner import run_from_config
from maxionbench.reports.paper_exports import generate_report_bundle


def _make_run(tmp_path: Path, *, overrides: dict[str, object] | None = None) -> Path:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 7,
        "repeats": 1,
        "no_retry": True,
        "output_dir": str(tmp_path / "run"),
        "quality_target": 0.8,
        "quality_targets": [0.8],
        "clients_read": 1,
        "clients_write": 0,
        "clients_grid": [1],
        "search_sweep": [{"hnsw_ef": 32}],
        "rpc_baseline_requests": 10,
        "sla_threshold_ms": 50.0,
        "vector_dim": 16,
        "num_vectors": 120,
        "num_queries": 10,
        "top_k": 10,
    }
    if overrides:
        cfg.update(overrides)
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)
    return run_from_config(cfg_path, cli_overrides=None)


def _assert_output_policy(
    meta: dict[str, object],
    *,
    expected_path_class: str,
    expected_milestone_id: str | None,
) -> None:
    policy = meta.get("output_policy")
    assert isinstance(policy, dict)
    assert policy.get("output_path_class") == expected_path_class
    assert policy.get("milestone_id") == expected_milestone_id
    assert isinstance(policy.get("mode"), str)
    assert isinstance(policy.get("resolved_out_dir"), str)
    if expected_path_class.startswith("milestones_"):
        assert isinstance(policy.get("milestone_root"), str)


def test_report_bundle_smoke(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    bundle = generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    assert bundle["figures"]
    assert bundle["tables"]
    assert any(path.suffix == ".png" for path in bundle["figures"])
    assert any(path.suffix == ".csv" for path in bundle["tables"])

    first_png = next(path for path in bundle["figures"] if path.suffix == ".png")
    first_meta = first_png.with_suffix(".meta.json")
    assert first_meta.exists()
    image = mpimg.imread(first_png)
    assert int(image.shape[0]) == 600
    assert int(image.shape[1]) == 600
    meta = json.loads(first_meta.read_text(encoding="utf-8"))
    assert meta["run_ids"]
    assert meta["config_fingerprints"]
    assert meta["dataset_bundles"]
    assert meta["seeds"]
    assert int(meta["font_size"]) == 16
    assert int(meta["panel_pixels"]) == 600
    assert int(meta["dpi"]) == 100
    _assert_output_policy(meta, expected_path_class="milestones_noncanonical", expected_milestone_id=None)

    assert (out_dir / "m8_deferred_note.md").exists()
    assert (out_dir / "m8_deferred_note.meta.json").exists()
    deferred_meta = json.loads((out_dir / "m8_deferred_note.meta.json").read_text(encoding="utf-8"))
    assert deferred_meta["dataset_bundles"]
    assert deferred_meta["seeds"]
    assert deferred_meta["run_ids"]
    assert deferred_meta["config_fingerprints"]
    _assert_output_policy(deferred_meta, expected_path_class="milestones_noncanonical", expected_milestone_id=None)
    assert (out_dir / "T1_environment_runtime_pinning.csv").exists()
    assert (out_dir / "T2_matched_quality_throughput_rhu.csv").exists()
    assert (out_dir / "T3_robustness_summary.csv").exists()
    assert (out_dir / "T4_decision_table.csv").exists()
    t1 = pd.read_csv(out_dir / "T1_environment_runtime_pinning.csv")
    t3 = pd.read_csv(out_dir / "T3_robustness_summary.csv")
    assert set(
        [
            "ground_truth_source",
            "ground_truth_metric",
            "ground_truth_k",
            "ground_truth_engine",
            "resource_cpu_vcpu_median",
            "resource_gpu_count_median",
            "resource_ram_gib_median",
            "resource_disk_tb_median",
            "rhu_rate_median",
            "w_c",
            "w_g",
            "w_r",
            "w_d",
            "c_ref_vcpu",
            "g_ref_gpu",
            "r_ref_gib",
            "d_ref_tb",
        ]
    ).issubset(t1.columns)
    assert str(t1.loc[0, "ground_truth_source"]) == "exact_topk"
    assert str(t1.loc[0, "ground_truth_metric"]) == "recall_at_10"
    assert int(t1.loc[0, "ground_truth_k"]) == 10
    assert str(t1.loc[0, "ground_truth_engine"]) == "numpy_exact"
    assert float(t1.loc[0, "resource_cpu_vcpu_median"]) >= 1.0
    assert float(t1.loc[0, "rhu_rate_median"]) > 0.0
    assert set(["p99_inflation_valid_rows", "p99_inflation_nan_rows", "p99_inflation_status"]).issubset(t3.columns)
    assert set(t3["p99_inflation_status"].dropna().astype(str).tolist()).issubset(
        {"computed_all_rows", "computed_partial_rows", "not_computable"}
    )

    summary_meta = json.loads((out_dir / "milestones_summary.meta.json").read_text(encoding="utf-8"))
    assert summary_meta["run_ids"]
    assert summary_meta["config_fingerprints"]
    _assert_output_policy(summary_meta, expected_path_class="milestones_noncanonical", expected_milestone_id=None)


def test_report_cli_smoke(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures_cli"
    code = cli_main(["report", "--input", str(run_dir.parent), "--mode", "final", "--out", str(out_dir)])
    assert code == 0
    assert any(out_dir.glob("*.png"))
    assert (out_dir / "F5_deferred_note.md").exists()
    assert (out_dir / "F5_deferred_note.meta.json").exists()
    summary_meta = json.loads((out_dir / "final_summary.meta.json").read_text(encoding="utf-8"))
    _assert_output_policy(summary_meta, expected_path_class="final", expected_milestone_id=None)


def test_report_cli_milestone_id_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = cli_main(["report", "--input", str(run_dir.parent), "--mode", "milestones", "--milestone-id", "M3"])
    assert code == 0
    out_dir = tmp_path / "artifacts" / "figures" / "milestones" / "M3"
    assert any(out_dir.glob("*.png"))
    assert (out_dir / "m8_deferred_note.md").exists()
    assert (out_dir / "m8_deferred_note.meta.json").exists()
    first_png_path = next(out_dir.glob("*.png"))
    first_meta_path = first_png_path.with_suffix(".meta.json")
    first_meta = json.loads(first_meta_path.read_text(encoding="utf-8"))
    _assert_output_policy(first_meta, expected_path_class="milestones_mx", expected_milestone_id="M3")


def test_report_meta_sidecars_all_include_output_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = cli_main(["report", "--input", str(run_dir.parent), "--mode", "milestones", "--milestone-id", "M4"])
    assert code == 0
    out_dir = tmp_path / "artifacts" / "figures" / "milestones" / "M4"
    meta_paths = sorted(out_dir.glob("*.meta.json"))
    assert meta_paths
    for meta_path in meta_paths:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        _assert_output_policy(payload, expected_path_class="milestones_mx", expected_milestone_id="M4")


def test_report_cli_rejects_invalid_milestone_id(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    with pytest.raises(ValueError, match="must match `M<integer>`"):
        cli_main(["report", "--input", str(run_dir.parent), "--mode", "milestones", "--milestone-id", "3"])


def test_report_cli_rejects_out_and_milestone_id_combination(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    with pytest.raises(ValueError, match="either --out or --milestone-id"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                str(tmp_path / "figures_cli_combo"),
                "--milestone-id",
                "M2",
            ]
        )


def test_report_cli_rejects_missing_out_for_final_mode(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    with pytest.raises(ValueError, match="--out is required when --mode final"):
        cli_main(["report", "--input", str(run_dir.parent), "--mode", "final"])


def test_report_bundle_rejects_non_mx_milestone_output_path(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    with pytest.raises(ValueError, match="milestones/Mx"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=Path("artifacts/figures/milestones"), mode="milestones")

    with pytest.raises(ValueError, match="must start with an `Mx` directory"):
        generate_report_bundle(
            input_dir=run_dir.parent,
            out_dir=Path("artifacts/figures/milestones/not_a_milestone"),
            mode="milestones",
        )

    with pytest.raises(ValueError, match="milestones/Mx"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                "artifacts/figures/milestones",
            ]
        )


def test_report_preflight_legacy_stage_timing_has_migration_hint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["setup_elapsed_s", "warmup_elapsed_s", "measure_elapsed_s", "export_elapsed_s"])
    legacy.to_parquet(results_path, index=False)

    with pytest.raises(RuntimeError, match="migrate-stage-timing"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=tmp_path / "figures_legacy", mode="milestones")

    with pytest.raises(RuntimeError, match="migrate-stage-timing"):
        cli_main(["report", "--input", str(run_dir.parent), "--mode", "milestones", "--out", str(tmp_path / "cli_legacy")])


def test_report_preflight_legacy_resource_profile_has_hint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    results_path = run_dir / "results.parquet"
    frame = pd.read_parquet(results_path)
    legacy = frame.drop(columns=["resource_cpu_vcpu", "resource_gpu_count", "resource_ram_gib", "resource_disk_tb", "rhu_rate"])
    legacy.to_parquet(results_path, index=False)

    with pytest.raises(RuntimeError, match="RHU resource profile"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=tmp_path / "figures_legacy_resource", mode="milestones")

    with pytest.raises(RuntimeError, match="RHU resource profile"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                str(tmp_path / "cli_legacy_resource"),
            ]
        )


def test_report_preflight_legacy_ground_truth_metadata_has_hint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("ground_truth_source", None)
    metadata.pop("ground_truth_metric", None)
    metadata.pop("ground_truth_k", None)
    metadata.pop("ground_truth_engine", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ground truth metadata"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=tmp_path / "figures_legacy_ground_truth", mode="milestones")

    with pytest.raises(RuntimeError, match="ground truth metadata"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                str(tmp_path / "cli_legacy_ground_truth"),
            ]
        )


def test_report_preflight_legacy_hardware_runtime_metadata_has_hint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("hardware_runtime", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hardware/runtime summary"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=tmp_path / "figures_legacy_hardware_runtime", mode="milestones")

    with pytest.raises(RuntimeError, match="hardware/runtime summary"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                str(tmp_path / "cli_legacy_hardware_runtime"),
            ]
        )


def test_report_preflight_legacy_rtt_baseline_request_profile_has_hint(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("rtt_baseline_request_profile", None)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="RTT baseline request-profile"):
        generate_report_bundle(input_dir=run_dir.parent, out_dir=tmp_path / "figures_legacy_rtt_profile", mode="milestones")

    with pytest.raises(RuntimeError, match="RTT baseline request-profile"):
        cli_main(
            [
                "report",
                "--input",
                str(run_dir.parent),
                "--mode",
                "milestones",
                "--out",
                str(tmp_path / "cli_legacy_rtt_profile"),
            ]
        )


def test_report_bundle_rejects_not_computable_robustness_inflation(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        overrides={
            "scenario": "s3_churn_smooth",
            "dataset_bundle": "D3",
            "dataset_hash": "synthetic-d3-v1",
            "clients_read": 8,
            "clients_write": 2,
            "clients_grid": [8],
            "sla_threshold_ms": 120.0,
            "num_vectors": 500,
            "num_queries": 30,
            "lambda_req_s": 200.0,
            "s3_read_rate": 160.0,
            "s3_insert_rate": 20.0,
            "s3_update_rate": 10.0,
            "s3_delete_rate": 10.0,
            "maintenance_interval_s": 15.0,
            "s3_max_events": 120,
            "d3_k_clusters": 48,
            "d3_num_tenants": 20,
            "d3_num_acl_buckets": 8,
            "d3_num_time_buckets": 12,
            "d3_beta_tenant": 0.75,
            "d3_beta_acl": 0.7,
            "d3_beta_time": 0.65,
            "d3_seed": 9,
            "allow_missing_s3_baseline": True,
        },
    )
    with pytest.raises(RuntimeError, match="robustness inflation is not computable"):
        generate_report_bundle(
            input_dir=run_dir.parent,
            out_dir=tmp_path / "figures_not_computable",
            mode="milestones",
        )
