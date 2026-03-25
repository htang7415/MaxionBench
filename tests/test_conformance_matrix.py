from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from maxionbench.conformance import matrix as conformance_matrix_mod
from maxionbench.conformance.matrix import main as conformance_matrix_main
from maxionbench.conformance.matrix import run_conformance_matrix
from maxionbench.conformance.provenance import conformance_provenance_path


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
    assert "stdout_path" in frame.columns
    assert "stderr_path" in frame.columns
    assert Path(str(frame.iloc[0]["stdout_path"])).exists()
    assert Path(str(frame.iloc[0]["stderr_path"])).exists()
    assert rows[0].stdout_path is not None
    assert rows[0].stderr_path is not None
    assert Path(rows[0].stdout_path).exists()
    assert Path(rows[0].stderr_path).exists()
    stdout_text = Path(rows[0].stdout_path).read_text(encoding="utf-8")
    assert '"event": "conformance_adapter_context"' in stdout_text
    assert '"event": "conformance_test_start"' in stdout_text
    assert '"event": "conformance_test_end"' in stdout_text
    assert (out_dir / "conformance_matrix.json").exists()
    provenance_path = conformance_provenance_path(out_dir / "conformance_matrix.csv")
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["matrix_path"] == str((out_dir / "conformance_matrix.csv").resolve())


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


def test_conformance_matrix_timeout_writes_partial_adapter_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_dir = tmp_path / "cfg_timeout"
    cfg_dir.mkdir(parents=True, exist_ok=True)
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

    def _raise_timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(
            cmd=["python", "-m", "maxionbench.conformance.run"],
            timeout=1.0,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(conformance_matrix_mod.subprocess, "run", _raise_timeout)
    out_dir = tmp_path / "out_timeout"
    rows = run_conformance_matrix(config_dir=cfg_dir, out_dir=out_dir, timeout_s=1.0)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "timeout"
    assert row.stdout_path is not None
    assert row.stderr_path is not None
    assert Path(row.stdout_path).read_text(encoding="utf-8") == "partial stdout"
    assert Path(row.stderr_path).read_text(encoding="utf-8") == "partial stderr"


def test_conformance_matrix_main_filters_to_selected_adapters(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg_filter"
    cfg_dir.mkdir(parents=True, exist_ok=True)
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
    (cfg_dir / "qdrant.json").write_text(
        json.dumps(
            {
                "adapter": "qdrant",
                "adapter_options_json": "{\"host\":\"127.0.0.1\",\"port\":6333}",
                "collection": "conformance",
                "dimension": 4,
                "metric": "ip",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_filter"
    code = conformance_matrix_main(
        [
            "--config-dir",
            str(cfg_dir),
            "--out-dir",
            str(out_dir),
            "--timeout-s",
            "20",
            "--adapters",
            "mock",
        ]
    )

    assert code == 0
    frame = pd.read_csv(out_dir / "conformance_matrix.csv")
    assert frame["adapter"].tolist() == ["mock"]
