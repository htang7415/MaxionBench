from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
