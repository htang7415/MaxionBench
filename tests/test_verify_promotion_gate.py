from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_promotion_gate import verify_promotion_gate


def test_verify_promotion_gate_passes_for_strict_ready_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": ["qdrant", "pgvector"],
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": 2,
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
        "required_adapters": ["qdrant", "pgvector"],
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 1,
        "errors": [{"message": "adapter `qdrant` failed conformance"}],
        "conformance_rows": 2,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert summary["ready_for_promotion"] is False
    assert summary["reasons"]


def test_verify_promotion_gate_cli_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": False,
        "required_adapters": [],
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": False,
        "error_count": 2,
        "errors": [{"message": "missing required adapters"}],
        "conformance_rows": 0,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code = cli_main(["verify-promotion-gate", "--strict-readiness-summary", str(summary_path), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["reasons"]


def test_verify_promotion_gate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_promotion_gate(strict_readiness_summary_path=tmp_path / "missing.json")


def test_verify_promotion_gate_rejects_allow_nonpass_mode(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": ["qdrant", "pgvector"],
        "allow_nonpass_status": True,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": 2,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "allow_nonpass_status=true" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_require_mock_pass(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": ["qdrant", "pgvector"],
        "allow_nonpass_status": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": 2,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_require_mock_pass_false(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": ["qdrant", "pgvector"],
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": 2,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass=false" in " ".join(summary["reasons"])
