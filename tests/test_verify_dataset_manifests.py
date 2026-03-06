from __future__ import annotations

from pathlib import Path
import shutil

import yaml

from maxionbench.tools import verify_dataset_manifests as manifests_mod


def test_verify_dataset_manifest_dir_passes_for_repo_manifests() -> None:
    summary = manifests_mod.verify_dataset_manifest_dir(Path("maxionbench/datasets/manifests"))
    assert summary["pass"] is True
    assert int(summary["error_count"]) == 0
    assert summary["checked_bundles"] == ["D1", "D2", "D3", "D4"]


def test_verify_dataset_manifest_dir_fails_when_required_manifest_missing(tmp_path: Path) -> None:
    src = Path("maxionbench/datasets/manifests")
    dst = tmp_path / "manifests"
    shutil.copytree(src, dst)
    (dst / "d3.yaml").unlink()

    summary = manifests_mod.verify_dataset_manifest_dir(dst)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    assert any("missing manifest for D3" in item["message"] for item in summary["errors"])


def test_verify_dataset_manifest_dir_fails_for_bad_checksum(tmp_path: Path) -> None:
    src = Path("maxionbench/datasets/manifests")
    dst = tmp_path / "manifests"
    shutil.copytree(src, dst)
    d4_path = dst / "d4.yaml"
    payload = yaml.safe_load(d4_path.read_text(encoding="utf-8"))
    payload["source_checksum_sha256"] = "bad"
    d4_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = manifests_mod.verify_dataset_manifest_dir(dst)
    assert summary["pass"] is False
    assert int(summary["error_count"]) >= 1
    assert any("source_checksum_sha256" in item["message"] for item in summary["errors"])


def test_verify_dataset_manifest_dir_fails_for_bad_optional_cache_checksum(tmp_path: Path) -> None:
    src = Path("maxionbench/datasets/manifests")
    dst = tmp_path / "manifests"
    shutil.copytree(src, dst)
    d3_path = dst / "d3.yaml"
    payload = yaml.safe_load(d3_path.read_text(encoding="utf-8"))
    payload["cache_sha256_dataset_path"] = "bad"
    d3_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    summary = manifests_mod.verify_dataset_manifest_dir(dst)
    assert summary["pass"] is False
    assert any("cache_sha256_dataset_path" in item["message"] for item in summary["errors"])


def test_verify_dataset_manifests_cli_returns_nonzero_on_fail(tmp_path: Path) -> None:
    src = Path("maxionbench/datasets/manifests")
    dst = tmp_path / "manifests"
    shutil.copytree(src, dst)
    d1_path = dst / "d1.yaml"
    payload = yaml.safe_load(d1_path.read_text(encoding="utf-8"))
    payload.pop("source_version", None)
    d1_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    code = manifests_mod.main(["--manifest-dir", str(dst), "--json"])
    assert code == 2
