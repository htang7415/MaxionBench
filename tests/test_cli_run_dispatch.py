from __future__ import annotations

import json

import pytest

from maxionbench.cli import main as cli_main
from maxionbench.conformance import matrix as conformance_matrix_mod
from maxionbench.orchestration import runner as runner_mod
from maxionbench.orchestration import run_matrix as run_matrix_mod
from maxionbench.tools import download_d1 as download_d1_mod
from maxionbench.tools import download_datasets as download_datasets_mod
from maxionbench.tools import execute_run_matrix as execute_run_matrix_mod
from maxionbench.tools import precompute_text_embeddings as precompute_text_embeddings_mod
from maxionbench.tools import preprocess_frames_portable as preprocess_frames_portable_mod
from maxionbench.tools import preprocess_datasets as preprocess_datasets_mod
from maxionbench.tools import verify_branch_protection as verify_branch_mod
from maxionbench.tools import verify_conformance_configs as verify_conformance_configs_mod
from maxionbench.tools import verify_d3_calibration as verify_d3_calibration_mod
from maxionbench.tools import verify_dataset_manifests as verify_dataset_manifests_mod
from maxionbench.tools import verify_engine_readiness as verify_engine_readiness_mod
from maxionbench.tools import verify_pins as verify_pins_mod
from maxionbench.tools import verify_promotion_gate as verify_promotion_gate_mod
from maxionbench.tools import wait_adapter as wait_adapter_mod
from maxionbench.tools import pre_run_gate as pre_run_gate_mod


def test_cli_run_dispatches_readiness_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 31

    monkeypatch.setattr(runner_mod, "main", _fake_main)
    code = cli_main(
        [
            "run",
            "--config",
            "configs/scenarios_portable/s1_single_hop.yaml",
            "--seed",
            "42",
            "--repeats",
            "3",
            "--no-retry",
            "--output-dir",
            "artifacts/runs/dispatch",
            "--d3-params",
            "artifacts/calibration/d3_params.yaml",
            "--enforce-readiness",
            "--conformance-matrix",
            "artifacts/conformance/conformance_matrix.csv",
            "--behavior-dir",
            "docs/behavior",
            "--allow-gpu-unavailable",
        ]
    )
    assert code == 31
    assert captured["argv"] == [
        "--config",
        "configs/scenarios_portable/s1_single_hop.yaml",
        "--seed",
        "42",
        "--repeats",
        "3",
        "--no-retry",
        "--output-dir",
        "artifacts/runs/dispatch",
        "--d3-params",
        "artifacts/calibration/d3_params.yaml",
        "--enforce-readiness",
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
        "--allow-gpu-unavailable",
    ]


def test_cli_run_dispatches_budget_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 32

    monkeypatch.setattr(runner_mod, "main", _fake_main)
    code = cli_main(
        [
            "run",
            "--config",
            "configs/scenarios_portable/s1_single_hop.yaml",
            "--budget",
            "b1",
        ]
    )
    assert code == 32
    assert captured["argv"] == [
        "--config",
        "configs/scenarios_portable/s1_single_hop.yaml",
        "--budget",
        "b1",
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
    ]


def test_cli_verify_branch_protection_dispatches_optional_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 37

    monkeypatch.setattr(verify_branch_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-branch-protection",
            "--repo",
            "owner/repo",
            "--branch",
            "main",
            "--timeout-s",
            "12.5",
            "--required-check",
            "report-preflight / report_preflight",
            "--include-drift-check",
            "--include-strict-readiness-check",
            "--include-publish-bundle-check",
            "--json",
        ]
    )
    assert code == 37
    assert captured["argv"] == [
        "--repo",
        "owner/repo",
        "--branch",
        "main",
        "--timeout-s",
        "12.5",
        "--required-check",
        "report-preflight / report_preflight",
        "--include-drift-check",
        "--include-strict-readiness-check",
        "--include-publish-bundle-check",
        "--json",
    ]


def test_cli_conformance_matrix_omits_empty_adapters_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 39

    monkeypatch.setattr(conformance_matrix_mod, "main", _fake_main)
    code = cli_main(
        [
            "conformance-matrix",
            "--config-dir",
            "configs/conformance",
            "--out-dir",
            "artifacts/conformance",
            "--timeout-s",
            "45.5",
        ]
    )
    assert code == 39
    assert captured["argv"] == [
        "--config-dir",
        "configs/conformance",
        "--out-dir",
        "artifacts/conformance",
        "--timeout-s",
        "45.5",
    ]


def test_cli_conformance_matrix_passes_nonempty_adapters_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 40

    monkeypatch.setattr(conformance_matrix_mod, "main", _fake_main)
    code = cli_main(
        [
            "conformance-matrix",
            "--config-dir",
            "configs/conformance",
            "--out-dir",
            "artifacts/conformance",
            "--timeout-s",
            "45.5",
            "--adapters",
            "mock,qdrant",
        ]
    )
    assert code == 40
    assert captured["argv"] == [
        "--config-dir",
        "configs/conformance",
        "--out-dir",
        "artifacts/conformance",
        "--timeout-s",
        "45.5",
        "--adapters",
        "mock,qdrant",
    ]


def test_cli_wait_adapter_dispatches_config_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 49

    monkeypatch.setattr(wait_adapter_mod, "main", _fake_main)
    code = cli_main(
        [
            "wait-adapter",
            "--config",
            "configs/scenarios_portable/s1_single_hop.yaml",
            "--timeout-s",
            "45",
            "--poll-interval-s",
            "2",
            "--json",
        ]
    )
    assert code == 49
    assert captured["argv"] == [
        "--timeout-s",
        "45.0",
        "--poll-interval-s",
        "2.0",
        "--config",
        "configs/scenarios_portable/s1_single_hop.yaml",
        "--json",
    ]


def test_cli_download_d1_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 51

    monkeypatch.setattr(download_d1_mod, "main", _fake_main)
    code = cli_main(
        [
            "download-d1",
            "--dataset-name",
            "gist-960-euclidean",
            "--output",
            "dataset/raw/gist.hdf5",
            "--force",
            "--timeout-s",
            "99",
            "--json",
        ]
    )
    assert code == 51
    assert captured["argv"] == [
        "--dataset-name",
        "gist-960-euclidean",
        "--timeout-s",
        "99.0",
        "--output",
        "dataset/raw/gist.hdf5",
        "--force",
        "--json",
    ]


def test_cli_download_datasets_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 53

    monkeypatch.setattr(download_datasets_mod, "main", _fake_main)
    code = cli_main(
        [
            "download-datasets",
            "--root",
            "dataset",
            "--cache-dir",
            ".cache",
            "--datasets",
            "",
            "--crag-examples",
            "123",
            "--skip-d3",
            "--force",
            "--timeout-s",
            "77",
            "--json",
        ]
    )
    assert code == 53
    assert captured["argv"] == [
        "--root",
        "dataset",
        "--cache-dir",
        ".cache",
        "--datasets",
        "",
        "--crag-examples",
        "123",
        "--timeout-s",
        "77.0",
        "--skip-d3",
        "--force",
        "--json",
    ]


def test_cli_download_datasets_dispatches_dataset_filter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 54

    monkeypatch.setattr(download_datasets_mod, "main", _fake_main)
    code = cli_main(
        [
            "download-datasets",
            "--root",
            "dataset",
            "--cache-dir",
            ".cache",
            "--datasets",
            "scifact,fiqa,crag,frames",
        ]
    )
    assert code == 54
    assert captured["argv"] == [
        "--root",
        "dataset",
        "--cache-dir",
        ".cache",
        "--datasets",
        "scifact,fiqa,crag,frames",
        "--crag-examples",
        "500",
        "--timeout-s",
        "60.0",
    ]


def test_cli_preprocess_datasets_dispatches_ann_hdf5(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 55

    monkeypatch.setattr(preprocess_datasets_mod, "main", _fake_main)
    code = cli_main(
        [
            "preprocess-datasets",
            "ann-hdf5",
            "--input",
            "dataset/raw/d1.hdf5",
            "--out",
            "dataset/processed/D1/gist-960-euclidean",
            "--family",
            "ann-benchmarks",
            "--name",
            "gist-960-euclidean",
            "--metric",
            "euclidean",
            "--json",
        ]
    )
    assert code == 55
    assert captured["argv"] == [
        "ann-hdf5",
        "--out",
        "dataset/processed/D1/gist-960-euclidean",
        "--input",
        "dataset/raw/d1.hdf5",
        "--family",
        "ann-benchmarks",
        "--name",
        "gist-960-euclidean",
        "--metric",
        "euclidean",
        "--json",
    ]


def test_cli_preprocess_datasets_rejects_missing_required_flags() -> None:
    with pytest.raises(SystemExit):
        cli_main(["preprocess-datasets", "ann-hdf5", "--out", "dataset/processed/D1/gist-960-euclidean"])


def test_cli_preprocess_frames_portable_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 56

    monkeypatch.setattr(preprocess_frames_portable_mod, "main", _fake_main)
    code = cli_main(
        [
            "preprocess-frames-portable",
            "--frames-root",
            "dataset/raw/frames",
            "--kilt-root",
            "dataset/raw/kilt",
            "--out",
            "dataset/processed/frames_portable",
            "--same-page-negatives",
            "6",
            "--cross-question-negatives",
            "6",
            "--seed",
            "42",
            "--json",
        ]
    )
    assert code == 56
    assert captured["argv"] == [
        "--frames-root",
        "dataset/raw/frames",
        "--kilt-root",
        "dataset/raw/kilt",
        "--out",
        "dataset/processed/frames_portable",
        "--same-page-negatives",
        "6",
        "--cross-question-negatives",
        "6",
        "--seed",
        "42",
        "--json",
    ]


def test_cli_precompute_text_embeddings_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 58

    monkeypatch.setattr(precompute_text_embeddings_mod, "main", _fake_main)
    code = cli_main(
        [
            "precompute-text-embeddings",
            "--input",
            "dataset/processed/D4",
            "--model-id",
            "BAAI/bge-small-en-v1.5",
            "--batch-size",
            "24",
            "--device",
            "mps",
            "--max-length",
            "256",
            "--require-device",
            "mps",
            "--no-normalize",
            "--force",
            "--json",
        ]
    )
    assert code == 58
    assert captured["argv"] == [
        "--input",
        "dataset/processed/D4",
        "--model-id",
        "BAAI/bge-small-en-v1.5",
        "--batch-size",
        "24",
        "--device",
        "mps",
        "--max-length",
        "256",
        "--require-device",
        "mps",
        "--no-normalize",
        "--force",
        "--json",
    ]


def test_cli_run_matrix_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 59

    monkeypatch.setattr(run_matrix_mod, "main", _fake_main)
    code = cli_main(
        [
            "run-matrix",
            "--scenario-config-dir",
            "configs/scenarios_portable",
            "--engine-config-dir",
            "configs/engines_portable",
            "--out-dir",
            "artifacts/run_matrix/portable_b0",
            "--output-root",
            "artifacts/runs/portable/b0",
            "--budget",
            "b0",
            "--lane",
            "cpu",
            "--json",
        ]
    )
    assert code == 59
    assert captured["argv"] == [
        "--scenario-config-dir",
        "configs/scenarios_portable",
        "--engine-config-dir",
        "configs/engines_portable",
        "--out-dir",
        "artifacts/run_matrix/portable_b0",
        "--output-root",
        "artifacts/runs/portable/b0",
        "--lane",
        "cpu",
        "--budget",
        "b0",
        "--json",
    ]


def test_cli_execute_run_matrix_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 60

    monkeypatch.setattr(execute_run_matrix_mod, "main", _fake_main)
    code = cli_main(
        [
            "execute-run-matrix",
            "--matrix",
            "artifacts/run_matrix/portable_b0/run_matrix.json",
            "--lane",
            "cpu",
            "--budget",
            "b0",
            "--seed",
            "42",
            "--skip-completed",
            "--continue-on-failure",
            "--engine-filter",
            "faiss-cpu,qdrant",
            "--scenario-filter",
            "s1_single_hop,s2_streaming_memory",
            "--template-filter",
            "s1_single_hop__bge-small-en-v1-5",
            "--max-runs",
            "7",
            "--deadline-hours",
            "24",
            "--adapter-timeout-s",
            "90",
            "--poll-interval-s",
            "2",
            "--json",
        ]
    )
    assert code == 60
    assert captured["argv"] == [
        "--matrix",
        "artifacts/run_matrix/portable_b0/run_matrix.json",
        "--lane",
        "cpu",
        "--adapter-timeout-s",
        "90.0",
        "--poll-interval-s",
        "2.0",
        "--budget",
        "b0",
        "--seed",
        "42",
        "--skip-completed",
        "--continue-on-failure",
        "--engine-filter",
        "faiss-cpu,qdrant",
        "--scenario-filter",
        "s1_single_hop,s2_streaming_memory",
        "--template-filter",
        "s1_single_hop__bge-small-en-v1-5",
        "--max-runs",
        "7",
        "--deadline-hours",
        "24.0",
        "--json",
    ]


def test_cli_verify_conformance_configs_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 57

    monkeypatch.setattr(verify_conformance_configs_mod, "main", _fake_main)
    code = cli_main(["verify-conformance-configs", "--config-dir", "configs/conformance", "--allow-gpu-unavailable", "--json"])
    assert code == 57
    assert captured["argv"] == ["--config-dir", "configs/conformance", "--allow-gpu-unavailable", "--json"]


def test_cli_verify_d3_calibration_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 59

    monkeypatch.setattr(verify_d3_calibration_mod, "main", _fake_main)
    code = cli_main(["verify-d3-calibration", "--d3-params", "artifacts/calibration/d3_params.yaml", "--strict", "--json"])
    assert code == 59
    assert captured["argv"] == ["--d3-params", "artifacts/calibration/d3_params.yaml", "--min-vectors", "10000000", "--strict", "--json"]


def test_cli_verify_engine_readiness_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 61

    monkeypatch.setattr(verify_engine_readiness_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            "artifacts/conformance/conformance_matrix.csv",
            "--behavior-dir",
            "docs/behavior",
            "--allow-gpu-unavailable",
            "--allow-nonpass-status",
            "--require-mock-pass",
            "--target-adapter",
            "qdrant",
            "--json",
        ]
    )
    assert code == 61
    assert captured["argv"] == [
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
        "--allow-gpu-unavailable",
        "--allow-nonpass-status",
        "--require-mock-pass",
        "--target-adapter",
        "qdrant",
        "--json",
    ]


def test_cli_pre_run_gate_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 63

    monkeypatch.setattr(pre_run_gate_mod, "main", _fake_main)
    code = cli_main(
        [
            "pre-run-gate",
            "--config",
            "configs/scenarios_portable/s1_single_hop.yaml",
            "--conformance-matrix",
            "artifacts/conformance/conformance_matrix.csv",
            "--behavior-dir",
            "docs/behavior",
            "--allow-gpu-unavailable",
            "--json",
        ]
    )
    assert code == 63
    assert captured["argv"] == [
        "--config",
        "configs/scenarios_portable/s1_single_hop.yaml",
        "--conformance-matrix",
        "artifacts/conformance/conformance_matrix.csv",
        "--behavior-dir",
        "docs/behavior",
        "--allow-gpu-unavailable",
        "--json",
    ]


def test_cli_verify_pins_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 65

    monkeypatch.setattr(verify_pins_mod, "main", _fake_main)
    code = cli_main(["verify-pins", "--config-dir", "configs/scenarios_portable", "--json"])
    assert code == 65
    assert captured["argv"] == ["--config-dir", "configs/scenarios_portable", "--json"]


def test_cli_verify_dataset_manifests_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 67

    monkeypatch.setattr(verify_dataset_manifests_mod, "main", _fake_main)
    code = cli_main(["verify-dataset-manifests", "--manifest-dir", "maxionbench/datasets/manifests", "--json"])
    assert code == 67
    assert captured["argv"] == ["--manifest-dir", "maxionbench/datasets/manifests", "--json"]


def test_cli_verify_promotion_gate_dispatches_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 69

    monkeypatch.setattr(verify_promotion_gate_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            "artifacts/conformance_strict/engine_readiness_summary.json",
            "--conformance-matrix",
            "artifacts/conformance_strict/conformance_matrix.csv",
            "--json",
        ]
    )
    assert code == 69
    assert captured["argv"] == [
        "--strict-readiness-summary",
        "artifacts/conformance_strict/engine_readiness_summary.json",
        "--conformance-matrix",
        "artifacts/conformance_strict/conformance_matrix.csv",
        "--json",
    ]


def test_cli_verify_promotion_gate_dispatches_portable_flags(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 70

    monkeypatch.setattr(verify_promotion_gate_mod, "main", _fake_main)
    code = cli_main(
        [
            "verify-promotion-gate",
            "--portable-results",
            "artifacts/runs/portable/b0",
            "--from-budget",
            "b0",
            "--out-candidates",
            "artifacts/run_matrix/portable_b0/promotion_candidates.json",
            "--json",
        ]
    )
    assert code == 70
    assert captured["argv"] == [
        "--portable-results",
        "artifacts/runs/portable/b0",
        "--from-budget",
        "b0",
        "--out-candidates",
        "artifacts/run_matrix/portable_b0/promotion_candidates.json",
        "--json",
    ]
