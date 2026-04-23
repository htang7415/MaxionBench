from __future__ import annotations

import os
from pathlib import Path
import tempfile

from maxionbench.tools import submit_portable as submit_mod


def test_submit_portable_builds_executes_and_checks_promotion(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, object]] = []
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")

    fake_matrix = submit_mod.RunMatrix(
        repo_root=str(tmp_path),
        generated_config_dir=str((tmp_path / "artifacts" / "run_matrix" / "portable_b0" / "generated_configs").resolve()),
        output_root=str((tmp_path / "artifacts" / "runs" / "portable" / "b0").resolve()),
        budget_level="b0",
        cpu_rows=[],
        gpu_rows=[],
        selected_engines=["qdrant"],
        selected_templates=["s1_single_hop__bge-small-en-v1-5.yaml"],
        lane="cpu",
    )

    def _fake_build_run_matrix(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("build", dict(kwargs)))
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_matrix.json").write_text("{}", encoding="utf-8")
        return fake_matrix

    def _fake_execute_run_matrix(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("execute", dict(kwargs)))
        return {"completed_rows": 2, "failed_rows": 0, "selected_rows": 2}

    def _fake_verify_portable_promotion_gate(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("promotion", dict(kwargs)))
        return {"pass": True, "ready_for_promotion": True}

    monkeypatch.setattr(submit_mod, "services_up", lambda **kwargs: calls.append(("services", dict(kwargs))) or {"healthy": True})
    monkeypatch.setattr(submit_mod, "build_run_matrix", _fake_build_run_matrix)
    monkeypatch.setattr(submit_mod, "execute_run_matrix", _fake_execute_run_matrix)
    monkeypatch.setattr(submit_mod, "verify_portable_promotion_gate", _fake_verify_portable_promotion_gate)

    summary = submit_mod.submit_portable(
        budget="b0",
        repo_root=tmp_path,
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
        seed=7,
    )

    assert summary["budget"] == "b0"
    assert summary["matrix_rows"] == {"cpu": 0, "gpu": 0, "all": 0}
    assert summary["execution"]["completed_rows"] == 2
    assert summary["promotion"]["pass"] is True
    assert calls[0][0] == "build"
    assert calls[1][0] == "services"
    assert calls[1][1]["services"] == ["qdrant"]
    assert calls[2][0] == "execute"
    assert calls[3][0] == "promotion"
    assert calls[2][1]["budget"] == "b0"
    assert calls[2][1]["skip_completed"] is True
    assert calls[3][1]["from_budget"] == "b0"


def test_submit_portable_skips_promotion_for_b2(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    fake_matrix = submit_mod.RunMatrix(
        repo_root=str(tmp_path),
        generated_config_dir=str((tmp_path / "artifacts" / "run_matrix" / "portable_b2" / "generated_configs").resolve()),
        output_root=str((tmp_path / "artifacts" / "runs" / "portable" / "b2").resolve()),
        budget_level="b2",
        cpu_rows=[],
        gpu_rows=[],
        selected_engines=["pgvector"],
        selected_templates=[],
        lane="cpu",
    )

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(submit_mod, "services_up", lambda **kwargs: calls.append(("services", dict(kwargs))) or {"healthy": True})
    monkeypatch.setattr(submit_mod, "build_run_matrix", lambda **kwargs: fake_matrix)
    monkeypatch.setattr(submit_mod, "execute_run_matrix", lambda **kwargs: calls.append(("execute", dict(kwargs))) or {"completed_rows": 1, "failed_rows": 0})

    called = {"promotion": False}

    def _fake_verify_portable_promotion_gate(**kwargs):  # type: ignore[no-untyped-def]
        called["promotion"] = True
        return {"pass": True}

    monkeypatch.setattr(submit_mod, "verify_portable_promotion_gate", _fake_verify_portable_promotion_gate)

    summary = submit_mod.submit_portable(
        budget="b2",
        repo_root=tmp_path,
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
    )

    assert "promotion" not in summary
    assert called["promotion"] is False
    assert calls[0][0] == "services"
    assert calls[0][1]["services"] == ["pgvector"]
    assert calls[1][0] == "execute"


def test_submit_portable_sets_local_lancedb_scratch_envs_when_unset(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    fake_matrix = submit_mod.RunMatrix(
        repo_root=str(tmp_path),
        generated_config_dir=str((tmp_path / "artifacts" / "run_matrix" / "portable_b0" / "generated_configs").resolve()),
        output_root=str((tmp_path / "artifacts" / "runs" / "portable" / "b0").resolve()),
        budget_level="b0",
        cpu_rows=[],
        gpu_rows=[],
        selected_engines=[],
        selected_templates=[],
        lane="cpu",
    )

    monkeypatch.delenv("MAXIONBENCH_LANCEDB_INPROC_URI", raising=False)
    monkeypatch.delenv("MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI", raising=False)
    monkeypatch.setattr(submit_mod, "services_up", lambda **kwargs: {"healthy": True})
    monkeypatch.setattr(submit_mod, "build_run_matrix", lambda **kwargs: fake_matrix)
    monkeypatch.setattr(submit_mod, "execute_run_matrix", lambda **kwargs: {"completed_rows": 1, "failed_rows": 0})

    submit_mod.submit_portable(
        budget="b0",
        repo_root=tmp_path,
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
        verify_promotion=False,
    )

    scratch_root = Path(tempfile.gettempdir()).resolve() / "maxionbench" / tmp_path.name / "lancedb"
    assert os.environ["MAXIONBENCH_LANCEDB_INPROC_URI"] == str((scratch_root / "inproc").resolve())
    assert os.environ["MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI"] == str((scratch_root / "service").resolve())
