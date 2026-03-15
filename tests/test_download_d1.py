from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from maxionbench.tools import download_d1 as download_d1_mod


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def test_download_d1_dataset_writes_default_cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    def _fake_urlopen(url: str, timeout: float):
        assert url == "https://ann-benchmarks.com/deep-image-96-angular.hdf5"
        assert timeout == 30.0
        return _FakeResponse(b"hdf5-bytes")

    monkeypatch.setattr(download_d1_mod, "urlopen", _fake_urlopen)
    summary = download_d1_mod.download_d1_dataset(
        dataset_name="deep-image-96-angular",
        timeout_s=30.0,
        output_path=download_d1_mod.default_output_path(
            dataset_name="deep-image-96-angular",
            repo_root=repo_root,
        ),
    )

    target = repo_root / "data" / "d1" / "deep-image-96-angular.hdf5"
    assert target.exists()
    assert target.read_bytes() == b"hdf5-bytes"
    assert summary["output_path"] == str(target.resolve())
    assert summary["source"] == "download"


def test_download_d1_dataset_uses_cache_when_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "data" / "d1" / "deep-image-96-angular.hdf5"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"cached-hdf5")

    def _unexpected_urlopen(url: str, timeout: float):
        raise AssertionError("network download should not run on cache hit")

    monkeypatch.setattr(download_d1_mod, "urlopen", _unexpected_urlopen)
    summary = download_d1_mod.download_d1_dataset(
        dataset_name="deep-image-96-angular",
        output_path=target,
    )

    assert summary["source"] == "cache_hit"
    assert summary["output_path"] == str(target.resolve())


def test_download_d1_dataset_rejects_invalid_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset_name must contain only"):
        download_d1_mod.download_d1_dataset(dataset_name="../escape")
