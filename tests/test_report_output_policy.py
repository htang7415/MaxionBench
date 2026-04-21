from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.orchestration.runner import run_from_config
from maxionbench.reports.paper_exports import generate_report_bundle
from maxionbench.tools import report_output_policy as report_output_policy_mod
from maxionbench.tools.report_output_policy import inspect_report_output_policy


def _make_run(tmp_path: Path) -> Path:
    cfg = {
        "engine": "mock",
        "engine_version": "0.1.0",
        "scenario": "s1_ann_frontier",
        "dataset_bundle": "D1",
        "dataset_hash": "synthetic-d1-v1",
        "seed": 11,
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
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return run_from_config(cfg_path, cli_overrides=None)


def test_inspect_report_output_policy_cli_dispatches_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 17

    monkeypatch.setattr(report_output_policy_mod, "main", _fake_main)
    code = cli_main(
        [
            "inspect-report-output-policy",
            "--input",
            "artifacts/figures/milestones/M3",
            "--output",
            "artifacts/ci/report_output_policy_summary.json",
            "--strict",
            "--json",
        ]
    )
    assert code == 17
    assert captured["argv"] == [
        "--input",
        "artifacts/figures/milestones/M3",
        "--output",
        "artifacts/ci/report_output_policy_summary.json",
        "--strict",
        "--json",
    ]


def test_inspect_report_output_policy_passes_on_generated_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    summary = inspect_report_output_policy(out_dir)
    assert summary["pass"] is True
    assert int(summary["meta_file_count"]) > 0
    assert summary["no_meta_files"] is False
    assert summary["mixed_modes"] is False
    assert summary["mixed_output_path_classes"] is False
    assert summary["mixed_milestone_ids"] is False
    assert summary["mixed_resolved_out_dirs"] is False
    assert summary["mixed_milestone_roots"] is False
    assert summary["output_path_class_mismatch"] is False
    assert summary["milestone_id_mismatch"] is False
    assert summary["resolved_out_dir_mismatch"] is False
    assert summary["milestone_root_mismatch"] is False
    assert summary["expected_output_path_class"] == "milestones_noncanonical"
    assert summary["expected_milestone_id"] is None
    assert summary["expected_resolved_out_dir"] == str(out_dir.resolve())
    assert summary["resolved_out_dirs"] == [str(out_dir.resolve())]
    expected_milestone_root = str((Path("artifacts/figures/milestones")).resolve())
    assert summary["expected_milestone_root"] == expected_milestone_root
    assert summary["milestone_roots"] == [expected_milestone_root]
    assert summary["missing_output_policy_files"] == []
    assert summary["invalid_output_policy"] == []
    assert summary["invalid_json_files"] == []
    assert summary["output_path_class_counts"]["milestones_noncanonical"] == int(summary["meta_file_count"])

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is True


def test_inspect_report_output_policy_cli_writes_output_artifact(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    artifact_path = tmp_path / "report_output_policy_summary.json"
    code = cli_main(
        [
            "inspect-report-output-policy",
            "--input",
            str(out_dir),
            "--output",
            str(artifact_path),
            "--strict",
            "--json",
        ]
    )
    assert code == 0
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert int(payload["meta_file_count"]) > 0


def test_inspect_report_output_policy_strict_fails_on_missing_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.pop("output_policy", None)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1
    assert str(target) in parsed["missing_output_policy_files"]


def test_inspect_report_output_policy_strict_fails_on_invalid_json_sidecar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    target.write_text("{not-json}\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1
    assert str(target) in parsed["invalid_json_files"]


def test_inspect_report_output_policy_strict_fails_on_invalid_policy_shape(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    policy = payload.get("output_policy")
    assert isinstance(policy, dict)
    policy["output_path_class"] = "milestones_mx"
    policy["milestone_id"] = None
    payload["output_policy"] = policy
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1
    issues_by_path = {
        str(item["path"]): [str(msg) for msg in item.get("issues", [])]
        for item in parsed["invalid_output_policy"]
        if isinstance(item, dict) and "path" in item
    }
    assert str(target) in issues_by_path
    assert any("milestone_id" in message for message in issues_by_path[str(target)])


def test_inspect_report_output_policy_strict_fails_on_mixed_policy_classes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    policy = payload.get("output_policy")
    assert isinstance(policy, dict)
    policy["mode"] = "final"
    policy["output_path_class"] = "final"
    policy["milestone_id"] = None
    policy.pop("milestone_root", None)
    payload["output_policy"] = policy
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["mixed_modes"] is True
    assert parsed["mixed_output_path_classes"] is True


def test_inspect_report_output_policy_strict_fails_on_mixed_out_dir_and_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    policy = payload.get("output_policy")
    assert isinstance(policy, dict)
    policy["resolved_out_dir"] = "/tmp/other-output-dir"
    policy["milestone_root"] = "/tmp/other-milestone-root"
    payload["output_policy"] = policy
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["mixed_resolved_out_dirs"] is True
    assert parsed["mixed_milestone_roots"] is True


def test_inspect_report_output_policy_strict_fails_on_uniform_mismatched_out_dir_and_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    for target in sorted(out_dir.glob("*.meta.json")):
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        policy = payload.get("output_policy")
        assert isinstance(policy, dict)
        policy["resolved_out_dir"] = "/tmp/relocated-figures"
        policy["milestone_root"] = "/tmp/relocated-milestone-root"
        payload["output_policy"] = policy
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["mixed_resolved_out_dirs"] is False
    assert parsed["mixed_milestone_roots"] is False
    assert parsed["resolved_out_dir_mismatch"] is True
    assert parsed["milestone_root_mismatch"] is True


def test_inspect_report_output_policy_passes_on_canonical_milestone_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts" / "figures" / "milestones" / "M4"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    summary = inspect_report_output_policy(out_dir)
    assert summary["pass"] is True
    assert summary["expected_output_path_class"] == "milestones_mx"
    assert summary["expected_milestone_id"] == "M4"
    assert summary["output_path_class_counts"]["milestones_mx"] == int(summary["meta_file_count"])
    assert summary["milestone_ids"] == ["M4"]
    assert summary["output_path_class_mismatch"] is False
    assert summary["milestone_id_mismatch"] is False


def test_inspect_report_output_policy_strict_fails_on_uniform_output_class_mismatch_for_mx_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts" / "figures" / "milestones" / "M3"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    for target in sorted(out_dir.glob("*.meta.json")):
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        policy = payload.get("output_policy")
        assert isinstance(policy, dict)
        policy["output_path_class"] = "milestones_noncanonical"
        policy["milestone_id"] = None
        payload["output_policy"] = policy
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict", "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["mixed_output_path_classes"] is False
    assert parsed["mixed_milestone_ids"] is False
    assert parsed["expected_output_path_class"] == "milestones_mx"
    assert parsed["expected_milestone_id"] == "M3"
    assert parsed["output_path_class_mismatch"] is True
    assert parsed["milestone_id_mismatch"] is True


def test_inspect_report_output_policy_cli_plaintext_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict"])
    assert code == 0
    text = capsys.readouterr().out.strip()
    assert text.startswith("pass:")
    assert "meta files checked" in text
    assert "issue(s)" in text


def test_inspect_report_output_policy_cli_plaintext_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures"
    generate_report_bundle(input_dir=run_dir.parent, out_dir=out_dir, mode="milestones")

    target = sorted(out_dir.glob("*.meta.json"))[0]
    target.write_text("{not-json}\n", encoding="utf-8")

    code = cli_main(["inspect-report-output-policy", "--input", str(out_dir), "--strict"])
    assert code == 2
    text = capsys.readouterr().out.strip()
    assert text.startswith("fail:")
    assert "meta files checked" in text
    assert "issue(s)" in text


def test_report_then_inspect_output_policy_cli_sequence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "figures_cli_sequence"

    report_code = cli_main(
        [
            "report",
            "--input",
            str(run_dir.parent),
            "--mode",
            "milestones",
            "--out",
            str(out_dir),
        ]
    )
    assert report_code == 0

    inspect_code = cli_main(
        [
            "inspect-report-output-policy",
            "--input",
            str(out_dir),
            "--strict",
            "--json",
        ]
    )
    assert inspect_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is True
    assert int(payload["error_count"]) == 0
    assert int(payload["meta_file_count"]) > 0
    assert payload["no_meta_files"] is False
    assert payload["output_path_class_counts"]["milestones_noncanonical"] == int(payload["meta_file_count"])


def test_inspect_report_output_policy_strict_fails_when_no_sidecars(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)

    code = cli_main(["inspect-report-output-policy", "--input", str(empty_dir), "--strict", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is False
    assert payload["no_meta_files"] is True
    assert int(payload["meta_file_count"]) == 0
    assert int(payload["error_count"]) >= 1


def test_inspect_report_output_policy_non_strict_reports_failure_but_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_dir = tmp_path / "empty_non_strict"
    empty_dir.mkdir(parents=True, exist_ok=True)

    code = cli_main(["inspect-report-output-policy", "--input", str(empty_dir)])
    assert code == 0
    text = capsys.readouterr().out.strip()
    assert text.startswith("fail:")
    assert "meta files checked" in text
