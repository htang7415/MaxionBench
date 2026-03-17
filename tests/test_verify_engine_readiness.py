from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

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


def test_verify_engine_readiness_rejects_mixed_pass_fail_status_in_strict_mode(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "qdrant", "status": "fail"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=False,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("non-pass statuses in strict mode" in message for message in messages)


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


def test_verify_engine_readiness_allows_nonpass_faiss_gpu_with_flag_in_strict_mode(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS if adapter != "faiss-gpu"]
    rows.append({"adapter": "faiss-gpu", "status": "fail"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=True,
        allow_nonpass_status=False,
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


def test_verify_engine_readiness_rejects_empty_adapter_rows_even_with_allow_nonpass(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "timeout"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "", "status": "pass"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("empty adapter values" in message for message in messages)


def test_verify_engine_readiness_rejects_empty_status_rows_even_with_allow_nonpass(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "timeout"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "mock", "status": ""})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("empty status values" in message for message in messages)


def test_verify_engine_readiness_allows_mixed_status_when_allow_nonpass_enabled(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "qdrant", "status": "fail"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_verify_engine_readiness_target_adapter_ignores_unrelated_failures(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [
        {"adapter": "qdrant", "status": "pass"},
        {"adapter": "milvus", "status": "fail"},
        {"adapter": "faiss-gpu", "status": "fail"},
    ]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=False,
        target_adapter="qdrant",
    )
    assert summary["pass"] is True
    assert summary["target_adapter"] == "qdrant"
    assert summary["required_adapters"] == ["qdrant"]


def test_verify_engine_readiness_target_adapter_rejects_target_failure(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [
        {"adapter": "qdrant", "status": "fail"},
        {"adapter": "milvus", "status": "pass"},
    ]
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=False,
        target_adapter="qdrant",
    )
    assert summary["pass"] is False
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("qdrant" in message.lower() for message in messages)


def test_verify_engine_readiness_rejects_nonpass_row_outside_required_adapters_in_strict_mode(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "experimental-adapter", "status": "invalid_config"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=False,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    messages = [str(item.get("message", "")) for item in summary["errors"]]
    assert any("strict readiness requires pass-only statuses" in message for message in messages)


def test_verify_engine_readiness_allows_nonpass_row_outside_required_adapters_when_flag_enabled(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "pass"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "experimental-adapter", "status": "invalid_config"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
    )
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0


def test_verify_engine_readiness_requires_mock_pass_when_enabled(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "timeout"} for adapter in REQUIRED_ADAPTERS]
    rows.append({"adapter": "mock", "status": "fail"})
    _write_matrix(matrix_path, rows)

    summary = verify_engine_readiness(
        conformance_matrix_path=matrix_path,
        behavior_dir=behavior_dir,
        allow_gpu_unavailable=False,
        allow_nonpass_status=True,
        require_mock_pass=True,
    )
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    assert any("adapter `mock` has no pass status" in str(item.get("message", "")) for item in summary["errors"])


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


def test_verify_engine_readiness_cli_dispatch_supports_require_mock_pass(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    rows = [{"adapter": adapter, "status": "fail"} for adapter in REQUIRED_ADAPTERS if adapter != "faiss-gpu"]
    rows.append({"adapter": "mock", "status": "pass"})
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
            "--require-mock-pass",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is True
    assert parsed["require_mock_pass"] is True


def test_verify_engine_readiness_cli_scopes_to_target_adapter(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_matrix(
        matrix_path,
        [
            {"adapter": "qdrant", "status": "pass"},
            {"adapter": "milvus", "status": "fail"},
        ],
    )

    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            str(matrix_path),
            "--behavior-dir",
            str(behavior_dir),
            "--target-adapter",
            "qdrant",
            "--json",
        ]
    )
    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is True
    assert parsed["target_adapter"] == "qdrant"


def test_verify_engine_readiness_cli_missing_matrix_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)
    code = cli_main(
        [
            "verify-engine-readiness",
            "--conformance-matrix",
            str(tmp_path / "missing.csv"),
            "--behavior-dir",
            str(behavior_dir),
            "--json",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-engine-readiness failed:" in captured.err


def test_verify_engine_readiness_cli_invalid_matrix_columns_returns_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    behavior_dir = tmp_path / "behavior"
    shutil.copytree(Path("docs/behavior"), behavior_dir)

    matrix_path = tmp_path / "conformance_matrix.csv"
    pd.DataFrame([{"adapter": "qdrant", "exit_code": 0}]).to_csv(matrix_path, index=False)

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
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-engine-readiness failed:" in captured.err
