from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from maxionbench.conformance.matrix import main as conformance_matrix_main
from maxionbench.conformance.matrix import run_conformance_matrix


def test_conformance_matrix_smoke_with_mock_only(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "mock.json"
    cfg_path.write_text(
        json.dumps(
            {
                "adapter": "mock",
                "adapter_options_json": "{}",
                "collection": "conformance",
                "dimension": 4,
                "metric": "ip",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    rows = run_conformance_matrix(config_dir=cfg_dir, out_dir=out_dir, timeout_s=60.0)
    assert len(rows) == 1
    assert rows[0].adapter == "mock"
    assert rows[0].status == "pass"
    assert rows[0].exit_code == 0

    frame = pd.read_csv(out_dir / "conformance_matrix.csv")
    assert len(frame) == 1
    assert frame.iloc[0]["status"] == "pass"
    assert (out_dir / "conformance_matrix.json").exists()


def test_conformance_matrix_invalid_config_is_recorded(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg_invalid"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "bad.json").write_text(json.dumps({"collection": "oops"}), encoding="utf-8")

    out_dir = tmp_path / "out_invalid"
    rows = run_conformance_matrix(config_dir=cfg_dir, out_dir=out_dir, timeout_s=10.0)
    assert len(rows) == 1
    assert rows[0].status == "invalid_config"
    assert rows[0].exit_code == 2


def test_conformance_matrix_malformed_json_is_recorded_and_does_not_abort(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg_malformed"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "bad.json").write_text("{not-valid-json", encoding="utf-8")
    (cfg_dir / "mock.json").write_text(
        json.dumps(
            {
                "adapter": "mock",
                "adapter_options_json": "{}",
                "collection": "conformance",
                "dimension": 4,
                "metric": "ip",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_malformed"
    rows = run_conformance_matrix(config_dir=cfg_dir, out_dir=out_dir, timeout_s=20.0)
    assert len(rows) == 2
    statuses = sorted(row.status for row in rows)
    assert statuses == ["invalid_config", "pass"]
    invalid_rows = [row for row in rows if row.status == "invalid_config"]
    assert len(invalid_rows) == 1
    assert "bad.json" in invalid_rows[0].config_file


def test_conformance_matrix_missing_config_dir_raises_file_not_found(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_conformance_matrix(config_dir=missing_dir, out_dir=tmp_path / "out_missing", timeout_s=10.0)


def test_conformance_matrix_empty_config_dir_raises_file_not_found(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "empty_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match=r"No conformance config files \(\*\.json\) found"):
        run_conformance_matrix(config_dir=cfg_dir, out_dir=tmp_path / "out_empty", timeout_s=10.0)


def test_conformance_matrix_main_returns_2_with_actionable_error(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    code = conformance_matrix_main(
        [
            "--config-dir",
            str(tmp_path / "missing_cfg_dir"),
            "--out-dir",
            str(tmp_path / "out"),
            "--timeout-s",
            "1",
        ]
    )
    assert code == 2
    stderr = capsys.readouterr().err
    assert "conformance-matrix failed:" in stderr
    assert "does not exist" in stderr
