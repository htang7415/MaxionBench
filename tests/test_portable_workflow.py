from __future__ import annotations

import json
from pathlib import Path

from maxionbench.tools import portable_workflow as workflow_mod


def test_ensure_lancedb_inproc_uri_sets_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAXIONBENCH_LANCEDB_INPROC_URI", raising=False)
    uri = workflow_mod.ensure_lancedb_inproc_uri(repo_root=tmp_path)
    assert uri == "/tmp/maxionbench/lancedb/inproc"


def test_portable_setup_runs_services_and_conformance(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.delenv("MAXIONBENCH_LANCEDB_INPROC_URI", raising=False)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: calls.append(("services", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "conformance_matrix_main", lambda argv=None: calls.append(("conformance", list(argv or []))) or 0)

    summary = workflow_mod.portable_setup(repo_root=tmp_path)

    assert summary["mode"] == "maxionbench-setup"
    assert calls == [
        ("services", ["up", "--profile", "maxionbench", "--wait", "--json"]),
        (
            "conformance",
            [
                "--config-dir",
                "configs/conformance",
                "--out-dir",
                "artifacts/conformance",
                "--timeout-s",
                "30",
                "--adapters",
                "mock,faiss-cpu,lancedb-inproc,qdrant,pgvector",
            ],
        ),
    ]


def test_portable_setup_raises_when_services_fail(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MAXIONBENCH_LANCEDB_INPROC_URI", raising=False)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: 1)
    monkeypatch.setattr(workflow_mod, "conformance_matrix_main", lambda argv=None: 0)

    try:
        workflow_mod.portable_setup(repo_root=tmp_path)
        raise AssertionError("expected portable_setup to fail")
    except RuntimeError as exc:
        assert str(exc) == "MaxionBench services startup failed with exit code 1"


def test_portable_data_runs_curated_download_and_embedding_jobs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []
    hotpot_root = tmp_path / "dataset" / "D4" / "hotpotqa"
    beir_scifact = tmp_path / "dataset" / "D4" / "beir" / "scifact"
    beir_fiqa = tmp_path / "dataset" / "D4" / "beir" / "fiqa"
    crag_root = tmp_path / "dataset" / "D4" / "crag"
    hotpot_root.mkdir(parents=True)
    beir_scifact.mkdir(parents=True)
    beir_fiqa.mkdir(parents=True)
    crag_root.mkdir(parents=True)
    (hotpot_root / "hotpot_dev_distractor_v1.json").write_text("[]\n", encoding="utf-8")
    (crag_root / "crag_task_1_and_2_dev_v4.first_500.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(workflow_mod, "download_datasets_main", lambda argv=None: calls.append(("download", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_datasets_main", lambda argv=None: calls.append(("preprocess_datasets", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_hotpot_portable_main", lambda argv=None: calls.append(("hotpot", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "precompute_text_embeddings_main", lambda argv=None: calls.append(("embed", list(argv or []))) or 0)

    summary = workflow_mod.portable_data(repo_root=tmp_path)

    assert summary["mode"] == "maxionbench-data"
    assert calls[0] == ("download", ["--root", "dataset", "--cache-dir", ".cache", "--datasets", "scifact,fiqa,crag,hotpotqa"])
    assert calls[1] == (
        "preprocess_datasets",
        ["beir", "--input", "dataset/D4/beir/scifact", "--out", "dataset/processed/D4/beir/scifact", "--name", "scifact"],
    )
    assert calls[2] == (
        "preprocess_datasets",
        ["beir", "--input", "dataset/D4/beir/fiqa", "--out", "dataset/processed/D4/beir/fiqa", "--name", "fiqa"],
    )
    assert calls[3] == (
        "preprocess_datasets",
        ["crag", "--input", "dataset/D4/crag/crag_task_1_and_2_dev_v4.first_500.jsonl", "--out", "dataset/processed/D4/crag/small_slice"],
    )
    assert calls[4] == (
        "hotpot",
        ["--input", "dataset/D4/hotpotqa/hotpot_dev_distractor_v1.json", "--out", "dataset/processed/hotpot_portable"],
    )
    assert len([call for call in calls if call[0] == "embed"]) == 4
    assert summary["hotpot_maxionbench"]["status"] == "processed"
    assert len(summary["d4_preprocess_jobs"]) == 3


def test_portable_data_fails_when_hotpot_source_is_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    beir_scifact = tmp_path / "dataset" / "D4" / "beir" / "scifact"
    beir_fiqa = tmp_path / "dataset" / "D4" / "beir" / "fiqa"
    crag_root = tmp_path / "dataset" / "D4" / "crag"
    beir_scifact.mkdir(parents=True)
    beir_fiqa.mkdir(parents=True)
    crag_root.mkdir(parents=True)
    (crag_root / "crag_task_1_and_2_dev_v4.first_500.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(workflow_mod, "download_datasets_main", lambda argv=None: 0)
    monkeypatch.setattr(workflow_mod, "preprocess_datasets_main", lambda argv=None: 0)
    monkeypatch.setattr(workflow_mod, "precompute_text_embeddings_main", lambda argv=None: 0)

    try:
        workflow_mod.portable_data(repo_root=tmp_path)
        raise AssertionError("expected portable_data to fail")
    except FileNotFoundError as exc:
        assert "HotpotQA-MaxionBench source missing" in str(exc)


def test_portable_finalize_runs_report_archive_and_shutdown(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        workflow_mod,
        "generate_portable_report_bundle",
        lambda *, input_dir, out_dir, conformance_matrix_path, behavior_dir: calls.append(
            ("report", (str(input_dir), str(out_dir), str(conformance_matrix_path), str(behavior_dir)))
        ),
    )
    monkeypatch.setattr(workflow_mod, "archive_main", lambda argv=None: calls.append(("archive", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: calls.append(("services", list(argv or []))) or 0)

    summary = workflow_mod.portable_finalize(repo_root=tmp_path)

    assert summary["mode"] == "maxionbench-finalize"
    assert calls[0] == (
        "report",
        (
            str((tmp_path / "artifacts/runs/portable").resolve()),
            str((tmp_path / "artifacts/figures/final").resolve()),
            str((tmp_path / "artifacts/conformance/conformance_matrix.csv").resolve()),
            str((tmp_path / "docs/behavior").resolve()),
        ),
    )
    assert calls[1] == (
        "archive",
        [
            "--runs-dir",
            "artifacts/runs/portable",
            "--figures-dir",
            "artifacts/figures/final",
            "--hotpot-maxionbench-dir",
            "dataset/processed/hotpot_portable",
            "--conformance-dir",
            "artifacts/conformance",
        ],
    )
    assert calls[2] == ("services", ["down", "--profile", "maxionbench"])


def test_portable_workflow_main_emits_single_json_error_on_failure(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(workflow_mod, "portable_setup", lambda repo_root: (_ for _ in ()).throw(RuntimeError("boom")))

    code = workflow_mod.main(["setup", "--repo-root", str(tmp_path), "--json"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"error": "boom"}
