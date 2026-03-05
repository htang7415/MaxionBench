from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.tools.pre_run_gate import evaluate_pre_run_gate
from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS


def _write_config(path: Path, *, engine: str) -> None:
    payload = {
        "engine": engine,
        "scenario": "s1_ann_frontier",
        "no_retry": True,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_conformance_matrix(path: Path, *, include_faiss_gpu: bool = True) -> None:
    adapters = list(REQUIRED_ADAPTERS)
    if not include_faiss_gpu:
        adapters = [name for name in adapters if name != "faiss-gpu"]
    frame = pd.DataFrame([{"adapter": adapter, "status": "pass"} for adapter in adapters])
    frame.to_csv(path, index=False)


def test_pre_run_gate_skips_mock_engine_without_readiness_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mock.yaml"
    _write_config(cfg_path, engine="mock")
    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=tmp_path / "missing.csv",
        behavior_dir=tmp_path / "missing_behavior",
    )
    assert summary["pass"] is True
    assert summary["skipped"] is True


def test_pre_run_gate_fails_real_engine_without_full_readiness(tmp_path: Path) -> None:
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "qdrant", "status": "pass"}]).to_csv(matrix_path, index=False)

    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )
    assert summary["pass"] is False
    readiness = summary["readiness"]
    assert isinstance(readiness, dict)
    assert int(readiness["error_count"]) >= 1


def test_pre_run_gate_passes_real_engine_with_full_readiness(tmp_path: Path) -> None:
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_conformance_matrix(matrix_path)

    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )
    assert summary["pass"] is True
    assert summary["skipped"] is False


def test_pre_run_gate_cli_reports_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_conformance_matrix(matrix_path, include_faiss_gpu=False)

    code = cli_main(
        [
            "pre-run-gate",
            "--config",
            str(cfg_path),
            "--conformance-matrix",
            str(matrix_path),
            "--behavior-dir",
            str(behavior_dir),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is False
    assert payload["readiness"]["pass"] is False


def test_pre_run_gate_cli_allows_gpu_omission_flag(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_conformance_matrix(matrix_path, include_faiss_gpu=False)

    code = cli_main(
        [
            "pre-run-gate",
            "--config",
            str(cfg_path),
            "--conformance-matrix",
            str(matrix_path),
            "--behavior-dir",
            str(behavior_dir),
            "--allow-gpu-unavailable",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is True
