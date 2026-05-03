from __future__ import annotations

import bz2
import json
from pathlib import Path
import zipfile

import pytest

from maxionbench.tools import download_datasets as download_datasets_mod


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


def test_download_hotpotqa_writes_official_dev_distractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_json = tmp_path / "hotpot.json"
    source_json.write_text("[]\n", encoding="utf-8")

    def _fake_download_file(*, url: str, dest: Path, timeout_s: float, force: bool):
        del timeout_s, force
        assert url == download_datasets_mod.HOTPOTQA_DEV_DISTRACTOR_URL
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source_json.read_bytes())
        return {"path": str(dest), "source": "download"}

    monkeypatch.setattr(download_datasets_mod, "download_file", _fake_download_file)

    summary = download_datasets_mod.download_hotpotqa(root=tmp_path / "dataset", timeout_s=10.0, force=False)

    dest = tmp_path / "dataset" / "D4" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    assert summary["source"] == "download"
    assert dest.read_text(encoding="utf-8") == "[]\n"


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


def test_download_datasets_respects_dataset_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"beir": 0, "crag": 0, "hotpotqa": 0}

    def _fake_download_beir_subsets(*, root: Path, timeout_s: float, force: bool, subsets: tuple[str, ...] | list[str]):
        del root, timeout_s, force
        calls["beir"] += 1
        return {"subsets": list(subsets)}

    def _fake_download_crag(*, root: Path, timeout_s: float, force: bool, max_examples: int):
        del root, timeout_s, force, max_examples
        calls["crag"] += 1
        return {"ok": True}

    def _fake_download_hotpotqa(*, root: Path, timeout_s: float, force: bool):
        del root, timeout_s, force
        calls["hotpotqa"] += 1
        return {"ok": True}

    monkeypatch.setattr(download_datasets_mod, "download_beir_subsets", _fake_download_beir_subsets)
    monkeypatch.setattr(download_datasets_mod, "download_crag", _fake_download_crag)
    monkeypatch.setattr(download_datasets_mod, "download_hotpotqa", _fake_download_hotpotqa)

    summary = download_datasets_mod.download_datasets(
        root=tmp_path / "dataset",
        cache_dir=tmp_path / ".cache",
        crag_examples=500,
        datasets=["scifact", "fiqa", "crag", "hotpotqa"],
    )

    assert calls == {"beir": 1, "crag": 1, "hotpotqa": 1}
    assert summary["requested_datasets"] == ["crag", "fiqa", "hotpotqa", "scifact"]


def test_download_datasets_writes_portable_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    cache_dir = tmp_path / ".cache"

    monkeypatch.setattr(download_datasets_mod, "download_beir_subsets", lambda **kwargs: {"beir": "ok"})
    monkeypatch.setattr(download_datasets_mod, "download_crag", lambda **kwargs: {"crag": "ok"})
    monkeypatch.setattr(download_datasets_mod, "download_hotpotqa", lambda **kwargs: {"hotpotqa": "ok"})

    summary = download_datasets_mod.download_datasets(
        root=root,
        cache_dir=cache_dir,
        crag_examples=7,
        force=False,
        timeout_s=15.0,
    )

    manifest_path = root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "maxionbench-bootstrap"
    assert "D1" not in manifest
    assert "D2" not in manifest
    assert "D3" not in manifest
    assert manifest["D4"]["beir"] == ["scifact", "fiqa"]
    assert manifest["D4"]["crag_small_slice"] == "crag_task_1_and_2_dev_v4.first_7.jsonl"
    assert "d4_beir" in summary["fetched"]
    assert "d4_crag" in summary["fetched"]
    assert "d4_hotpotqa" in summary["fetched"]


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
