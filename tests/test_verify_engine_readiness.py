from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_engine_readiness import REQUIRED_ADAPTERS, verify_engine_readiness


def _write_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def test_verify_engine_readiness_passes_with_full_conformance_and_behavior_cards(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["behavior_cards_ok"] is True


def test_verify_engine_readiness_detects_failed_adapter_status(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    rows = [dict(item) for item in rows]
    rows.append({"adapter": "qdrant", "status": "fail"})
    rows = [item for item in rows if not (item["adapter"] == "qdrant" and item["status"] == "pass")]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("qdrant" in message.lower() for message in messages)


def test_verify_engine_readiness_allows_missing_faiss_gpu_with_flag(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS if adapter != "faiss-gpu"]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=True,
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_verify_engine_readiness_cli_reports_failure_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    (behavior_dir / "lancedb.md").write_text("# broken\n", encoding="utf-8")

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    _write_matrix(matrix_path, rows)

    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            str(matrix_path),
            "--behavior-dir",
            str(behavior_dir),
            "--json",
        ]
    )
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert int(parsed["error_count"]) >= 1


def test_verify_engine_readiness_allows_nonpass_status_when_flag_enabled(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "timeout"} for adapter in REQUIRED_ADAPTERS]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_verify_engine_readiness_cli_dispatch_supports_allow_nonpass_status(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "fail"} for adapter in REQUIRED_ADAPTERS if adapter != "faiss-gpu"]
    _write_matrix(matrix_path, rows)

    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            str(matrix_path),
            "--behavior-dir",
            str(behavior_dir),
            "--allow-gpu-unavailable",
            "--allow-nonpass-status",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is True
