from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from maxionbench.cli import main as cli_main
from maxionbench.conformance.provenance import conformance_provenance_path
from maxionbench.tools import pre_run_gate as pre_run_gate_mod
from maxionbench.tools.pre_run_gate import evaluate_pre_run_gate
from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS


def _write_config(
    path: Path,
    *,
    engine: str,
    scenario: str = "s1_ann_frontier",
    s5_require_hf_backend: bool = False,
) -> None:
    payload = {
        "engine": engine,
        "scenario": scenario,
        "no_retry": True,
    }
    if scenario == "s5_rerank":
        payload["s5_require_hf_backend"] = bool(s5_require_hf_backend)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_conformance_matrix(path: Path) -> None:
    adapters = list(REQUIRED_ADAPTERS)
    frame = pd.DataFrame([{"adapter": adapter, "status": "pass"} for adapter in adapters])
    frame.to_csv(path, index=False)


def _write_conformance_provenance(
    path: Path,
    *,
    container_runtime: str = "apptainer",
    container_image: str = "/shared/containers/maxionbench.sif",
) -> None:
    provenance_path = conformance_provenance_path(path)
    provenance_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-17T00:00:00+00:00",
                "container_runtime": container_runtime,
                "container_image": container_image,
                "python_executable": "/usr/bin/python",
                "hostname": "gpu-node-01",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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


def test_pre_run_gate_passes_real_engine_with_target_readiness_only(tmp_path: Path) -> None:
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
    assert summary["pass"] is True
    readiness = summary["readiness"]
    assert isinstance(readiness, dict)
    assert int(readiness["error_count"]) == 0


def test_pre_run_gate_ignores_unrelated_failed_adapters(tmp_path: Path) -> None:
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame(
        [
            {"adapter": "qdrant", "status": "pass"},
            {"adapter": "pgvector", "status": "fail"},
        ]
    ).to_csv(matrix_path, index=False)

    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )
    assert summary["pass"] is True
    assert summary["readiness"]["target_adapter"] == "qdrant"


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
    _write_conformance_matrix(matrix_path)

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
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pass"] is True
    assert payload["readiness"]["target_adapter"] == "qdrant"


def test_pre_run_gate_cli_reports_target_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "qdrant", "status": "fail"}]).to_csv(matrix_path, index=False)

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


def test_pre_run_gate_cli_accepts_deprecated_gpu_omission_flag(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_conformance_matrix(matrix_path)

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


def test_pre_run_gate_rejects_s5_hf_requirement_when_env_or_deps_missing(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    cfg_path = tmp_path / "s5_mock.yaml"
    _write_config(
        cfg_path,
        engine="mock",
        scenario="s5_rerank",
        s5_require_hf_backend=True,
    )
    monkeypatch.setattr(pre_run_gate_mod, "_detect_gpu_count", lambda: 0)
    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=tmp_path / "missing.csv",
        behavior_dir=tmp_path / "missing_behavior",
    )
    assert summary["pass"] is False
    assert summary["reason"] == "s5 reranker runtime requirements not satisfied"
    runtime = summary["s5_reranker_runtime"]
    assert isinstance(runtime, dict)
    assert runtime["required"] is True
    assert runtime["pass"] is False
    assert runtime["gpu_count"] == 0
    assert any("MAXIONBENCH_ENABLE_HF_RERANKER" in msg for msg in runtime["errors"])
    assert any("at least one NVIDIA GPU must be visible" in msg for msg in runtime["errors"])


def test_pre_run_gate_accepts_s5_hf_requirement_when_runtime_flags_are_ready(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    cfg_path = tmp_path / "s5_mock_ready.yaml"
    _write_config(
        cfg_path,
        engine="mock",
        scenario="s5_rerank",
        s5_require_hf_backend=True,
    )
    monkeypatch.setenv("MAXIONBENCH_ENABLE_HF_RERANKER", "1")

    def _fake_find_spec(name: str):  # type: ignore[no-untyped-def]
        if name in {"torch", "transformers"}:
            return object()
        return None

    monkeypatch.setattr(pre_run_gate_mod.importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(pre_run_gate_mod, "_detect_gpu_count", lambda: 1)
    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=tmp_path / "missing.csv",
        behavior_dir=tmp_path / "missing_behavior",
    )
    assert summary["pass"] is True
    assert summary["skipped"] is True
    runtime = summary["s5_reranker_runtime"]
    assert runtime["required"] is True
    assert runtime["pass"] is True
    assert runtime["errors"] == []


def test_pre_run_gate_rejects_missing_conformance_provenance_for_container_runtime(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "qdrant", "status": "pass"}]).to_csv(matrix_path, index=False)

    monkeypatch.setenv("MAXIONBENCH_CONTAINER_RUNTIME", "apptainer")
    monkeypatch.setenv("MAXIONBENCH_CONTAINER_IMAGE", "/shared/containers/maxionbench.sif")
    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )
    assert summary["pass"] is False
    assert summary["reason"] == "conformance provenance validation failed"
    assert summary["conformance_provenance"]["pass"] is False


def test_pre_run_gate_accepts_matching_conformance_provenance_for_container_runtime(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    cfg_path = tmp_path / "qdrant.yaml"
    _write_config(cfg_path, engine="qdrant")

    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "qdrant", "status": "pass"}]).to_csv(matrix_path, index=False)
    _write_conformance_provenance(matrix_path)

    monkeypatch.setenv("MAXIONBENCH_CONTAINER_RUNTIME", "apptainer")
    monkeypatch.setenv("MAXIONBENCH_CONTAINER_IMAGE", "/shared/containers/maxionbench.sif")
    summary = evaluate_pre_run_gate(
        config_path=cfg_path,
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
    )
    assert summary["pass"] is True
    assert summary["conformance_provenance"]["pass"] is True
