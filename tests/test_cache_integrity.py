from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from maxionbench.datasets.cache_integrity import (
    normalize_sha256,
    resolve_expected_sha256,
    sha256_file,
    verify_file_sha256,
)


def test_sha256_file_and_verify_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    payload = b"maxionbench-cache-check"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_file(path) == expected
    verify_file_sha256(path=path, expected_sha256=expected, label="sample")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_file_sha256(path=path, expected_sha256=("0" * 64), label="sample")


def test_normalize_sha256_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="sha256"):
        normalize_sha256("not-a-hash", field_label="test")


def test_resolve_expected_sha256_prefers_config_then_manifest() -> None:
    config = {"dataset_path_sha256": "a" * 64}
    manifest = {"cache_sha256_dataset_path": "b" * 64}
    got = resolve_expected_sha256(
        config_payload=config,
        manifest_payload=manifest,
        config_key="dataset_path_sha256",
        manifest_key="cache_sha256_dataset_path",
        label="d1",
    )
    assert got == "a" * 64

    got_manifest = resolve_expected_sha256(
        config_payload={},
        manifest_payload=manifest,
        config_key="dataset_path_sha256",
        manifest_key="cache_sha256_dataset_path",
        label="d1",
    )
    assert got_manifest == "b" * 64
