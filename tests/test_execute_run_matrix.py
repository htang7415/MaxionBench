from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from maxionbench.tools import execute_run_matrix as execute_mod


def _write_config(path: Path, *, engine: str, scenario: str, output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "engine": engine,
                "scenario": scenario,
                "output_dir": str(output_dir),
                "no_retry": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_matrix(path: Path, *, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "repo_root": str(Path(".").resolve()),
                "generated_config_dir": str(path.parent.resolve()),
                "output_root": "artifacts/runs/test",
                "cpu_rows": rows,
                "gpu_rows": [],
                "selected_engines": sorted({str(row["engine"]) for row in rows}),
                "selected_templates": sorted({str(row["template_name"]) for row in rows}),
                "lane": "cpu",
                "budget_level": "b0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_execute_run_matrix_skips_completed_and_filters_rows(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg_a = tmp_path / "generated" / "a.yaml"
    cfg_b = tmp_path / "generated" / "b.yaml"
    out_a = tmp_path / "runs" / "a"
    out_b = tmp_path / "runs" / "b"
    _write_config(cfg_a, engine="qdrant", scenario="s1_single_hop", output_dir=out_a)
    _write_config(cfg_b, engine="faiss-cpu", scenario="s2_streaming_memory", output_dir=out_b)
    (out_a / "run_status.json").parent.mkdir(parents=True, exist_ok=True)
    (out_a / "run_status.json").write_text(
        json.dumps({"status": "success", "timestamp_utc": "2026-04-21T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    matrix_path = tmp_path / "run_matrix.json"
    _write_matrix(
        matrix_path,
        rows=[
            {
                "group": "cpu",
                "config_path": str(cfg_a.resolve()),
                "engine": "qdrant",
                "scenario": "s1_single_hop",
                "dataset_bundle": "D4",
                "template_name": "s1_single_hop__bge-small-en-v1-5.yaml",
            },
            {
                "group": "cpu",
                "config_path": str(cfg_b.resolve()),
                "engine": "faiss-cpu",
                "scenario": "s2_streaming_memory",
                "dataset_bundle": "D4",
                "template_name": "s2_streaming_memory__bge-small-en-v1-5.yaml",
            },
        ],
    )

    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def _fake_wait_for_adapter(*, adapter_name: str, adapter_options: dict, timeout_s: float, poll_interval_s: float) -> dict:
        calls.append(("wait", adapter_name, {"timeout_s": timeout_s, "poll_interval_s": poll_interval_s}))
        return {"ready": True}

    def _fake_run_from_config(config_path: Path, cli_overrides: dict[str, object] | None = None) -> Path:
        calls.append(("run", str(config_path), dict(cli_overrides or {})))
        return Path(config_path)

    monkeypatch.setattr(execute_mod, "wait_for_adapter", _fake_wait_for_adapter)
    monkeypatch.setattr(execute_mod, "run_from_config", _fake_run_from_config)

    summary = execute_mod.execute_run_matrix(
        matrix_path=matrix_path,
        lane="cpu",
        budget="b0",
        seed=42,
        skip_completed=True,
        engine_filter={"qdrant", "faiss-cpu"},
        scenario_filter={"s1_single_hop", "s2_streaming_memory"},
        max_runs=2,
    )

    assert summary["selected_rows"] == 2
    assert summary["skipped_rows"] == 1
    assert summary["completed_rows"] == 1
    assert summary["failed_rows"] == 0
    assert calls == [("run", str(cfg_b.resolve()), {"budget_level": "b0", "seed": 42})]


def test_execute_run_matrix_can_continue_after_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg_a = tmp_path / "generated" / "a.yaml"
    cfg_b = tmp_path / "generated" / "b.yaml"
    _write_config(cfg_a, engine="qdrant", scenario="s1_single_hop", output_dir=tmp_path / "runs" / "a")
    _write_config(cfg_b, engine="faiss-cpu", scenario="s1_single_hop", output_dir=tmp_path / "runs" / "b")
    matrix_path = tmp_path / "run_matrix.json"
    _write_matrix(
        matrix_path,
        rows=[
            {
                "group": "cpu",
                "config_path": str(cfg_a.resolve()),
                "engine": "qdrant",
                "scenario": "s1_single_hop",
                "dataset_bundle": "D4",
                "template_name": "s1_single_hop__bge-small-en-v1-5.yaml",
            },
            {
                "group": "cpu",
                "config_path": str(cfg_b.resolve()),
                "engine": "faiss-cpu",
                "scenario": "s1_single_hop",
                "dataset_bundle": "D4",
                "template_name": "s1_single_hop__bge-base-en-v1-5.yaml",
            },
        ],
    )

    seen: list[str] = []

    def _fake_wait_for_adapter(*, adapter_name: str, adapter_options: dict, timeout_s: float, poll_interval_s: float) -> dict:
        del adapter_options, timeout_s, poll_interval_s
        seen.append(f"wait:{adapter_name}")
        return {"ready": True}

    def _fake_run_from_config(config_path: Path, cli_overrides: dict[str, object] | None = None) -> Path:
        del cli_overrides
        seen.append(str(Path(config_path).name))
        if Path(config_path).name == "a.yaml":
            raise RuntimeError("simulated failure")
        return Path(config_path)

    monkeypatch.setattr(execute_mod, "wait_for_adapter", _fake_wait_for_adapter)
    monkeypatch.setattr(execute_mod, "run_from_config", _fake_run_from_config)

    summary = execute_mod.execute_run_matrix(
        matrix_path=matrix_path,
        lane="cpu",
        continue_on_failure=True,
    )

    assert summary["completed_rows"] == 1
    assert summary["failed_rows"] == 1
    assert len(summary["failures"]) == 1
    assert "a.yaml" in summary["failures"][0]["config_path"]
    assert seen[0] == "wait:qdrant"


def test_execute_run_matrix_raises_without_continue_on_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "generated" / "a.yaml"
    _write_config(cfg, engine="faiss-cpu", scenario="s1_single_hop", output_dir=tmp_path / "runs" / "a")
    matrix_path = tmp_path / "run_matrix.json"
    _write_matrix(
        matrix_path,
        rows=[
            {
                "group": "cpu",
                "config_path": str(cfg.resolve()),
                "engine": "faiss-cpu",
                "scenario": "s1_single_hop",
                "dataset_bundle": "D4",
                "template_name": "s1_single_hop__bge-small-en-v1-5.yaml",
            }
        ],
    )

    def _fake_run_from_config(config_path: Path, cli_overrides: dict[str, object] | None = None) -> Path:
        del config_path, cli_overrides
        raise RuntimeError("boom")

    monkeypatch.setattr(execute_mod, "run_from_config", _fake_run_from_config)

    with pytest.raises(RuntimeError, match="failed_rows"):
        execute_mod.execute_run_matrix(matrix_path=matrix_path, lane="cpu", continue_on_failure=False)


def test_deadline_warning_projects_over_budget(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    summary = {
        "selected_rows": 10,
        "completed_rows": 1,
        "skipped_rows": 0,
        "warnings": [],
    }
    monkeypatch.setattr(execute_mod.time, "perf_counter", lambda: 3600.0)

    execute_mod._append_deadline_warning(summary=summary, started=0.0, deadline_hours=1.0)

    assert summary["warnings"]
    assert summary["warnings"][0]["type"] == "deadline_projection"
