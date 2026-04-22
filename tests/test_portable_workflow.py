from __future__ import annotations

import json
from pathlib import Path

from maxionbench.tools import portable_workflow as workflow_mod


def test_ensure_lancedb_service_uri_sets_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI", raising=False)
    uri = workflow_mod.ensure_lancedb_service_uri(repo_root=tmp_path)
    assert uri == str((tmp_path / "artifacts/lancedb/service").resolve())


def test_portable_setup_runs_services_and_conformance(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.delenv("MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI", raising=False)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: calls.append(("services", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "conformance_matrix_main", lambda argv=None: calls.append(("conformance", list(argv or []))) or 0)

    summary = workflow_mod.portable_setup(repo_root=tmp_path)

    assert summary["mode"] == "portable-setup"
    assert calls == [
        ("services", ["up", "--profile", "portable", "--wait", "--json"]),
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
                "mock,faiss-cpu,lancedb-inproc,lancedb-service,qdrant,pgvector",
            ],
        ),
    ]


def test_portable_setup_raises_when_services_fail(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI", raising=False)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: 1)
    monkeypatch.setattr(workflow_mod, "conformance_matrix_main", lambda argv=None: 0)

    try:
        workflow_mod.portable_setup(repo_root=tmp_path)
        raise AssertionError("expected portable_setup to fail")
    except RuntimeError as exc:
        assert str(exc) == "portable services startup failed with exit code 1"


def test_portable_data_runs_curated_download_and_embedding_jobs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []
    frames_root = tmp_path / "dataset" / "D4" / "frames"
    kilt_root = tmp_path / "dataset" / "D4" / "kilt"
    beir_scifact = tmp_path / "dataset" / "D4" / "beir" / "scifact"
    beir_fiqa = tmp_path / "dataset" / "D4" / "beir" / "fiqa"
    crag_root = tmp_path / "dataset" / "D4" / "crag"
    frames_root.mkdir(parents=True)
    kilt_root.mkdir(parents=True)
    beir_scifact.mkdir(parents=True)
    beir_fiqa.mkdir(parents=True)
    crag_root.mkdir(parents=True)
    (frames_root / "questions.jsonl").write_text('{"query_id":"q1","text":"q","gold_evidence":[{"page_id":"p1","text":"e"}]}\n', encoding="utf-8")
    (kilt_root / "pages.jsonl").write_text('{"page_id":"p1","text":"e"}\n', encoding="utf-8")
    (crag_root / "crag_task_1_and_2_dev_v4.first_500.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(workflow_mod, "download_datasets_main", lambda argv=None: calls.append(("download", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_datasets_main", lambda argv=None: calls.append(("preprocess_datasets", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_frames_portable_main", lambda argv=None: calls.append(("frames", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "precompute_text_embeddings_main", lambda argv=None: calls.append(("embed", list(argv or []))) or 0)

    summary = workflow_mod.portable_data(repo_root=tmp_path)

    assert summary["mode"] == "portable-data"
    assert calls[0] == ("download", ["--root", "dataset", "--cache-dir", ".cache", "--datasets", "scifact,fiqa,crag,frames"])
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
        "frames",
        ["--frames-root", "dataset/D4/frames", "--kilt-root", "dataset/D4/kilt", "--out", "dataset/processed/frames_portable"],
    )
    assert len([call for call in calls if call[0] == "embed"]) == 4
    assert summary["frames_portable"]["status"] == "processed"
    assert len(summary["d4_preprocess_jobs"]) == 3


def test_portable_data_reports_manual_frames_and_kilt_inputs_without_failing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []
    beir_scifact = tmp_path / "dataset" / "D4" / "beir" / "scifact"
    beir_fiqa = tmp_path / "dataset" / "D4" / "beir" / "fiqa"
    crag_root = tmp_path / "dataset" / "D4" / "crag"
    beir_scifact.mkdir(parents=True)
    beir_fiqa.mkdir(parents=True)
    crag_root.mkdir(parents=True)
    (crag_root / "crag_task_1_and_2_dev_v4.first_500.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(workflow_mod, "download_datasets_main", lambda argv=None: 0)
    monkeypatch.setattr(workflow_mod, "preprocess_datasets_main", lambda argv=None: calls.append(("preprocess_datasets", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_frames_portable_main", lambda argv=None: calls.append(("frames", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "precompute_text_embeddings_main", lambda argv=None: calls.append(("embed", list(argv or []))) or 0)

    summary = workflow_mod.portable_data(repo_root=tmp_path)

    assert summary["frames_portable"]["status"] == "manual_required"
    assert len(summary["frames_portable"]["missing_inputs"]) == 2
    assert not any(call[0] == "frames" for call in calls)
    assert [call for call in calls if call[0] == "embed"] == [
        ("embed", ["--input", "dataset/processed/D4", "--model-id", "BAAI/bge-small-en-v1.5"]),
        ("embed", ["--input", "dataset/processed/D4", "--model-id", "BAAI/bge-base-en-v1.5"]),
    ]


def test_portable_finalize_runs_report_archive_and_shutdown(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        workflow_mod,
        "generate_portable_report_bundle",
        lambda *, input_dir, out_dir: calls.append(("report", (str(input_dir), str(out_dir)))),
    )
    monkeypatch.setattr(workflow_mod, "archive_main", lambda argv=None: calls.append(("archive", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "services_main", lambda argv=None: calls.append(("services", list(argv or []))) or 0)

    summary = workflow_mod.portable_finalize(repo_root=tmp_path)

    assert summary["mode"] == "portable-finalize"
    assert calls[0] == (
        "report",
        (
            str((tmp_path / "artifacts/runs/portable").resolve()),
            str((tmp_path / "artifacts/figures/final").resolve()),
        ),
    )
    assert calls[1] == (
        "archive",
        [
            "--runs-dir",
            "artifacts/runs/portable",
            "--figures-dir",
            "artifacts/figures/final",
            "--frames-portable-dir",
            "dataset/processed/frames_portable",
            "--conformance-dir",
            "artifacts/conformance",
        ],
    )
    assert calls[2] == ("services", ["down", "--profile", "portable"])


def test_portable_workflow_main_emits_single_json_error_on_failure(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(workflow_mod, "portable_setup", lambda repo_root: (_ for _ in ()).throw(RuntimeError("boom")))

    code = workflow_mod.main(["setup", "--repo-root", str(tmp_path), "--json"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"error": "boom"}
