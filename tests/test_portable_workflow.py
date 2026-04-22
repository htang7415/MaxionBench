from __future__ import annotations

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


def test_portable_data_runs_curated_download_and_embedding_jobs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(workflow_mod, "download_datasets_main", lambda argv=None: calls.append(("download", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "preprocess_frames_portable_main", lambda argv=None: calls.append(("frames", list(argv or []))) or 0)
    monkeypatch.setattr(workflow_mod, "precompute_text_embeddings_main", lambda argv=None: calls.append(("embed", list(argv or []))) or 0)

    summary = workflow_mod.portable_data(repo_root=tmp_path)

    assert summary["mode"] == "portable-data"
    assert calls[0] == ("download", ["--root", "dataset", "--cache-dir", ".cache", "--datasets", "scifact,fiqa,crag,frames"])
    assert calls[1] == (
        "frames",
        ["--frames-root", "dataset/raw/frames", "--kilt-root", "dataset/raw/kilt", "--out", "dataset/processed/frames_portable"],
    )
    assert len([call for call in calls if call[0] == "embed"]) == 4


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
