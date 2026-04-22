from __future__ import annotations

import bz2
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from maxionbench.tools import download_datasets as download_datasets_mod


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_download_ann_benchmarks_writes_requested_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path]] = []

    def _fake_download_d1_dataset(*, dataset_name: str, output_path: Path, force: bool, timeout_s: float):
        calls.append((dataset_name, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"hdf5")
        return {"dataset_name": dataset_name, "output_path": str(output_path), "force": force, "timeout_s": timeout_s}

    monkeypatch.setattr(download_datasets_mod, "download_d1_dataset", _fake_download_d1_dataset)
    root = tmp_path / "dataset"
    summary = download_datasets_mod.download_ann_benchmarks(root=root, timeout_s=12.0, force=False)

    assert list(summary.keys()) == [
        "D1/glove-100-angular.hdf5",
        "D1/sift-128-euclidean.hdf5",
        "D1/gist-960-euclidean.hdf5",
        "D2/deep-image-96-angular.hdf5",
    ]
    assert calls[0][0] == "glove-100-angular"
    assert calls[-1][0] == "deep-image-96-angular"
    assert (root / "D1" / "glove-100-angular.hdf5").exists()
    assert (root / "D2" / "deep-image-96-angular.hdf5").exists()


def test_download_beir_extracts_selected_subsets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archives: dict[str, Path] = {}
    for dataset in download_datasets_mod.BEIR_DATASETS:
        archive_path = tmp_path / f"{dataset}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"{dataset}/corpus.jsonl", '{"_id":"d1","text":"doc"}\n')
            archive.writestr(f"{dataset}/queries.jsonl", '{"_id":"q1","text":"query"}\n')
            archive.writestr(f"{dataset}/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
        archives[dataset] = archive_path

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool):
        del timeout_s, force
        dataset = Path(url).stem
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archives[dataset].read_bytes())
        return {"url": url, "path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)
    root = tmp_path / "dataset"
    summary = download_datasets_mod.download_beir(root=root, timeout_s=20.0, force=False)

    assert set(summary.keys()) == set(download_datasets_mod.BEIR_DATASETS)
    assert (root / "D4" / "beir" / "scifact" / "corpus.jsonl").exists()
    assert (root / "D4" / "beir" / "fiqa" / "queries.jsonl").exists()
    assert (root / "D4" / "beir" / "nfcorpus" / "qrels" / "test.tsv").exists()


def test_download_beir_subsets_filters_requested_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", '{"_id":"d1","text":"doc"}\n')
        archive.writestr("scifact/queries.jsonl", '{"_id":"q1","text":"query"}\n')
        archive.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool):
        del url, timeout_s, force
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive_path.read_bytes())
        return {"path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)
    root = tmp_path / "dataset"
    summary = download_datasets_mod.download_beir_subsets(
        root=root,
        timeout_s=20.0,
        force=False,
        subsets=("scifact",),
    )

    assert set(summary.keys()) == {"scifact"}
    assert (root / "D4" / "beir" / "scifact" / "corpus.jsonl").exists()
    assert not (root / "D4" / "beir" / "fiqa").exists()


def test_download_crag_writes_archive_and_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_bz2 = tmp_path / "crag_source.jsonl.bz2"
    with bz2.open(source_bz2, "wt", encoding="utf-8") as handle:
        for idx in range(4):
            handle.write(json.dumps({"id": idx, "query": f"q{idx}"}) + "\n")

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool):
        del url, timeout_s, force
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source_bz2.read_bytes())
        return {"path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)
    root = tmp_path / "dataset"
    summary = download_datasets_mod.download_crag(root=root, timeout_s=10.0, force=False, max_examples=3)

    archive_path = root / "D4" / "crag" / "crag_task_1_and_2_dev_v4.jsonl.bz2"
    slice_path = root / "D4" / "crag" / "crag_task_1_and_2_dev_v4.first_3.jsonl"
    assert archive_path.exists()
    assert slice_path.exists()
    assert summary["slice"]["examples"] == 3
    assert len([line for line in slice_path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 3


def test_prepare_frames_workspace_creates_manual_placeholder(tmp_path: Path) -> None:
    summary = download_datasets_mod.prepare_frames_workspace(root=tmp_path / "dataset")
    frames_root = tmp_path / "dataset" / "D4" / "frames"

    assert summary["source"] == "manual_required"
    assert frames_root.exists()
    assert (frames_root / "README.manual.txt").exists()


def test_prepare_kilt_workspace_creates_manual_placeholder(tmp_path: Path) -> None:
    summary = download_datasets_mod.prepare_kilt_workspace(root=tmp_path / "dataset")
    kilt_root = tmp_path / "dataset" / "D4" / "kilt"

    assert summary["source"] == "manual_required"
    assert kilt_root.exists()
    assert (kilt_root / "README.manual.txt").exists()


def test_download_file_sets_explicit_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if captured.get("used"):
                return b""
            captured["used"] = True
            return b"payload"

    def _fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["user_agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(download_datasets_mod, "urlopen", _fake_urlopen)
    dest = tmp_path / "file.bin"
    summary = download_datasets_mod.download_file(
        url="https://example.com/file.bin",
        dest=dest,
        timeout_s=15.0,
        force=True,
    )

    assert summary["source"] == "download"
    assert captured["url"] == "https://example.com/file.bin"
    assert captured["user_agent"] == "MaxionBench/0.1"
    assert captured["timeout"] == 15.0
    assert dest.read_bytes() == b"payload"


def test_download_bigann_yfcc_copies_generated_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    cache_dir = tmp_path / ".cache"
    commands: list[list[str]] = []
    deps_checked: list[bool] = []

    def _fake_stage_bigann_snapshot(
        *,
        snapshot_url: str,
        repo_dir: Path,
        cache_dir: Path,
        timeout_s: float,
        force: bool,
    ) -> dict[str, str]:
        del snapshot_url, cache_dir, timeout_s, force
        (repo_dir / "data" / "yfcc100M").mkdir(parents=True, exist_ok=True)
        (repo_dir / "data" / "yfcc100M" / "vectors.bin").write_bytes(b"abc")
        (repo_dir / "requirements_py3.10.txt").write_text("numpy\n", encoding="utf-8")
        return {"path": str(repo_dir), "source": "download"}

    def _fake_run(cmd: list[str], *, cwd: Path | None = None) -> None:
        del cwd
        commands.append(list(cmd))

    def _fake_ensure_bigann_runtime_dependencies() -> None:
        deps_checked.append(True)

    monkeypatch.setattr(download_datasets_mod, "stage_bigann_snapshot", _fake_stage_bigann_snapshot)
    monkeypatch.setattr(download_datasets_mod, "run", _fake_run)
    monkeypatch.setattr(download_datasets_mod, "_ensure_bigann_runtime_dependencies", _fake_ensure_bigann_runtime_dependencies)
    summary = download_datasets_mod.download_bigann_yfcc(root=root, cache_dir=cache_dir, timeout_s=30.0, force=False)

    assert (root / "D3" / "yfcc-10M" / "vectors.bin").exists()
    assert summary["source"] == "copied_from_bigann_snapshot"
    assert deps_checked == [True]
    assert not any("pip" in cmd for cmd in commands)
    assert any("create_dataset.py" in cmd for cmd in commands)


def test_find_existing_yfcc_dir_accepts_upstream_yfcc100m_name(tmp_path: Path) -> None:
    repo_dir = tmp_path / "big-ann-benchmarks"
    upstream_dir = repo_dir / "data" / "yfcc100M"
    upstream_dir.mkdir(parents=True)

    found = download_datasets_mod._find_existing_yfcc_dir(repo_dir)
    assert found == upstream_dir


def test_ensure_bigann_runtime_dependencies_reports_missing_dists(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_version(dist_name: str) -> str:
        if dist_name in {"docker", "scikit-learn"}:
            raise download_datasets_mod.importlib_metadata.PackageNotFoundError(dist_name)
        return "1.0"

    monkeypatch.setattr(download_datasets_mod.importlib_metadata, "version", _fake_version)

    with pytest.raises(RuntimeError, match="docker, scikit-learn"):
        download_datasets_mod._ensure_bigann_runtime_dependencies()


def test_download_bigann_yfcc_uses_existing_dst_cache_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    dst = root / "D3" / "yfcc-10M"
    dst.mkdir(parents=True)
    (dst / "vectors.bin").write_bytes(b"cached")

    def _fail_stage_bigann_snapshot(
        *,
        snapshot_url: str,
        repo_dir: Path,
        cache_dir: Path,
        timeout_s: float,
        force: bool,
    ) -> dict[str, str]:
        del snapshot_url, repo_dir, cache_dir, timeout_s, force
        raise AssertionError("stage_bigann_snapshot should not run when destination cache already exists")

    def _fail_run(cmd: list[str], *, cwd: Path | None = None) -> None:
        raise AssertionError("run() should not be called when destination cache already exists")

    monkeypatch.setattr(download_datasets_mod, "stage_bigann_snapshot", _fail_stage_bigann_snapshot)
    monkeypatch.setattr(download_datasets_mod, "run", _fail_run)

    summary = download_datasets_mod.download_bigann_yfcc(
        root=root,
        cache_dir=tmp_path / ".cache",
        timeout_s=30.0,
        force=False,
    )
    assert summary["source"] == "cache_hit"


def test_download_datasets_respects_dataset_filter_without_d1d2_or_d3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"ann": 0, "d3": 0, "beir": 0, "crag": 0, "frames": 0, "kilt": 0}

    def _fake_download_ann_benchmarks(*, root: Path, timeout_s: float, force: bool):
        del root, timeout_s, force
        calls["ann"] += 1
        return {"ok": True}

    def _fake_download_bigann_yfcc(*, root: Path, cache_dir: Path, timeout_s: float, force: bool, requirements_file: str | None = None):
        del root, cache_dir, timeout_s, force, requirements_file
        calls["d3"] += 1
        return {"ok": True}

    def _fake_download_beir_subsets(*, root: Path, timeout_s: float, force: bool, subsets: tuple[str, ...] | list[str]):
        del root, timeout_s, force
        calls["beir"] += 1
        return {"subsets": list(subsets)}

    def _fake_download_crag(*, root: Path, timeout_s: float, force: bool, max_examples: int):
        del root, timeout_s, force, max_examples
        calls["crag"] += 1
        return {"ok": True}

    def _fake_prepare_frames_workspace(*, root: Path):
        del root
        calls["frames"] += 1
        return {"ok": True}

    def _fake_prepare_kilt_workspace(*, root: Path):
        del root
        calls["kilt"] += 1
        return {"ok": True}

    monkeypatch.setattr(download_datasets_mod, "download_ann_benchmarks", _fake_download_ann_benchmarks)
    monkeypatch.setattr(download_datasets_mod, "download_bigann_yfcc", _fake_download_bigann_yfcc)
    monkeypatch.setattr(download_datasets_mod, "download_beir_subsets", _fake_download_beir_subsets)
    monkeypatch.setattr(download_datasets_mod, "download_crag", _fake_download_crag)
    monkeypatch.setattr(download_datasets_mod, "prepare_frames_workspace", _fake_prepare_frames_workspace)
    monkeypatch.setattr(download_datasets_mod, "prepare_kilt_workspace", _fake_prepare_kilt_workspace)

    summary = download_datasets_mod.download_datasets(
        root=tmp_path / "dataset",
        cache_dir=tmp_path / ".cache",
        crag_examples=500,
        datasets=["scifact", "fiqa", "crag", "frames"],
    )

    assert calls == {"ann": 0, "d3": 0, "beir": 1, "crag": 1, "frames": 1, "kilt": 1}
    assert summary["requested_datasets"] == ["crag", "fiqa", "frames", "scifact"]


def test_stage_bigann_snapshot_extracts_repo_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_source = tmp_path / "bigann_snapshot.tar.gz"
    payload_root = tmp_path / "payload" / download_datasets_mod.BIGANN_SNAPSHOT_ROOT_NAME
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / "create_dataset.py").write_text("print('ok')\n", encoding="utf-8")
    (payload_root / "benchmark").mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_source, "w:gz") as archive:
        archive.add(payload_root, arcname=download_datasets_mod.BIGANN_SNAPSHOT_ROOT_NAME)

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool) -> dict[str, str]:
        del url, timeout_s, force
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive_source.read_bytes())
        return {"url": "https://example.com/bigann.tar.gz", "path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)
    repo_dir = tmp_path / ".cache" / "big-ann-benchmarks"
    summary = download_datasets_mod.stage_bigann_snapshot(
        snapshot_url="https://example.com/bigann.tar.gz",
        repo_dir=repo_dir,
        cache_dir=tmp_path / ".cache",
        timeout_s=30.0,
        force=False,
    )

    assert summary["source"] == "download"
    assert (repo_dir / "create_dataset.py").exists()


def test_stage_bigann_snapshot_refreshes_incomplete_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_source = tmp_path / "bigann_snapshot.tar.gz"
    payload_root = tmp_path / "payload" / download_datasets_mod.BIGANN_SNAPSHOT_ROOT_NAME
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / "create_dataset.py").write_text("print('ok')\n", encoding="utf-8")
    (payload_root / "benchmark").mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_source, "w:gz") as archive:
        archive.add(payload_root, arcname=download_datasets_mod.BIGANN_SNAPSHOT_ROOT_NAME)

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool) -> dict[str, str]:
        del url, timeout_s, force
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive_source.read_bytes())
        return {"url": "https://example.com/bigann.tar.gz", "path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)
    repo_dir = tmp_path / ".cache" / "big-ann-benchmarks"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "partial.txt").write_text("partial\n", encoding="utf-8")

    summary = download_datasets_mod.stage_bigann_snapshot(
        snapshot_url="https://example.com/bigann.tar.gz",
        repo_dir=repo_dir,
        cache_dir=tmp_path / ".cache",
        timeout_s=30.0,
        force=False,
    )

    assert summary["source"] == "download"
    assert not (repo_dir / "partial.txt").exists()
    assert (repo_dir / "create_dataset.py").exists()


def test_download_datasets_writes_manifest_with_skip_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    cache_dir = tmp_path / ".cache"

    monkeypatch.setattr(download_datasets_mod, "download_ann_benchmarks", lambda **kwargs: {"d1_d2": "ok"})
    monkeypatch.setattr(download_datasets_mod, "download_bigann_yfcc", lambda **kwargs: {"d3": "ok"})
    monkeypatch.setattr(download_datasets_mod, "download_beir_subsets", lambda **kwargs: {"beir": "ok"})
    monkeypatch.setattr(download_datasets_mod, "download_crag", lambda **kwargs: {"crag": "ok"})
    monkeypatch.setattr(download_datasets_mod, "prepare_frames_workspace", lambda **kwargs: {"frames": "ok"})
    monkeypatch.setattr(download_datasets_mod, "prepare_kilt_workspace", lambda **kwargs: {"kilt": "ok"})

    summary = download_datasets_mod.download_datasets(
        root=root,
        cache_dir=cache_dir,
        crag_examples=7,
        skip_d1d2=False,
        skip_d3=True,
        skip_d4=False,
        force=False,
        timeout_s=15.0,
    )

    manifest_path = root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["D4"]["crag_small_slice"] == "crag_task_1_and_2_dev_v4.first_7.jsonl"
    assert manifest["D4"]["kilt_workspace"] == "D4/kilt/"
    assert "d1_d2" in summary["fetched"]
    assert "d3" not in summary["fetched"]
    assert "d4_beir" in summary["fetched"]
    assert "d4_crag" in summary["fetched"]
    assert "d4_frames" in summary["fetched"]
    assert "d4_kilt" in summary["fetched"]


def test_download_beir_rejects_invalid_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        download_datasets_mod.download_beir(root=tmp_path / "dataset", timeout_s=0.0, force=False)


def test_download_crag_rejects_invalid_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        download_datasets_mod.download_crag(
            root=tmp_path / "dataset",
            timeout_s=-1.0,
            force=False,
            max_examples=5,
        )
