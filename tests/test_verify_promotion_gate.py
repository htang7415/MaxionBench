from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_promotion_gate import verify_promotion_gate

STRICT_REQUIRED_ADAPTERS = [
    "qdrant",
    "milvus",
    "weaviate",
    "opensearch",
    "pgvector",
    "lancedb-service",
    "lancedb-inproc",
    "faiss-cpu",
    "faiss-gpu",
]
STRICT_REQUIRED_ADAPTER_COUNT = len(STRICT_REQUIRED_ADAPTERS)


def _write_matrix(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["adapter", "status"])
        writer.writeheader()
        for adapter, status in rows:
            writer.writerow({"adapter": adapter, "status": status})


def test_verify_promotion_gate_passes_for_strict_ready_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_fails_for_nonpassing_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": False,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 1,
        "errors": [{"message": "adapter `qdrant` failed conformance"}],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT - 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert summary["ready_for_promotion"] is False
    assert summary["reasons"]


def test_verify_promotion_gate_passes_for_gpu_omitted_strict_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required),
        "conformance_status_counts": {"pass": len(required)},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_rejects_gpu_omitted_nonpass_counts_without_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required) + 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "provide --conformance-matrix" in " ".join(summary["reasons"])


def test_verify_promotion_gate_passes_gpu_omitted_with_faiss_gpu_nonpass_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required) + 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in required] + [("mock", "pass"), ("faiss-gpu", "fail")]
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_rejects_gpu_omitted_nonpass_on_non_gpu_adapter(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required), "fail": 2},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in required if adapter != "qdrant"]
    rows.extend([("qdrant", "fail"), ("mock", "pass"), ("faiss-gpu", "fail")])
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "permits non-pass matrix rows only for `faiss-gpu`" in " ".join(summary["reasons"])


def test_verify_promotion_gate_passes_with_conformance_matrix_cross_check(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_matrix(matrix_path, [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "pass")])
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []
    assert summary["conformance_matrix_path"] == str(matrix_path.resolve())


def test_verify_promotion_gate_rejects_conformance_matrix_status_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mismatch_rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "pass")]
    mismatch_rows[-1] = (mismatch_rows[-1][0], "fail")
    _write_matrix(matrix_path, mismatch_rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "status_counts disagrees" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_required_adapter_missing_from_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS]
    # Replace one required adapter with mock to keep row/status counts unchanged.
    rows[-1] = ("mock", "pass")
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "required_adapters missing in conformance matrix" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_mock_pass_row_when_required(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_matrix(matrix_path, [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS])
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "requires mock pass" in " ".join(summary["reasons"])


def test_verify_promotion_gate_cli_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": False,
        "required_adapters": [],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": False,
        "error_count": 2,
        "errors": [{"message": "missing required adapters"}],
        "conformance_rows": 0,
        "conformance_status_counts": {"fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code = cli_main(["verify-promotion-gate", "--strict-readiness-summary", str(summary_path), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["reasons"]


def test_verify_promotion_gate_cli_missing_matrix_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            str(summary_path),
            "--conformance-matrix",
            str(tmp_path / "missing.csv"),
            "--json",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-promotion-gate failed:" in captured.err


def test_verify_promotion_gate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_promotion_gate(strict_readiness_summary_path=tmp_path / "missing.json")


def test_verify_promotion_gate_missing_matrix_file_raises(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=tmp_path / "missing.csv",
        )


def test_verify_promotion_gate_matrix_missing_required_columns_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path.write_text("adapter,exit_code\nqdrant,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_matrix_empty_status_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "")]
    _write_matrix(matrix_path, rows)
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_matrix_empty_adapter_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("", "pass")]
    _write_matrix(matrix_path, rows)
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_cli_invalid_matrix_columns_returns_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path.write_text("adapter,exit_code\nqdrant,0\n", encoding="utf-8")
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            str(summary_path),
            "--conformance-matrix",
            str(matrix_path),
            "--json",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-promotion-gate failed:" in captured.err


def test_verify_promotion_gate_rejects_allow_nonpass_mode(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": True,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "allow_nonpass_status=true" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_require_mock_pass(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_require_mock_pass_false(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass=false" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_allow_gpu_unavailable(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "allow_gpu_unavailable" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_required_adapters_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "required_adapters mismatch" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_non_string_required_adapters(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS[:-1] + [7],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "required_adapters must contain only strings" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_conformance_status_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "conformance_status_counts" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_nonpass_conformance_status_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT - 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "includes non-pass rows" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_conformance_status_count_sum_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "row count mismatch" in " ".join(summary["reasons"])
