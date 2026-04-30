"""Verify dataset manifest coverage and pinned metadata fields."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_MANIFESTS = {
    "D4": "d4.yaml",
}
_REQUIRED_SIZE_KEYS = (
    "approx_bytes_dataset",
    "approx_bytes_engine",
    "approx_bytes_temp",
)
_OPTIONAL_CACHE_CHECKSUM_KEYS = (
    "cache_sha256_d4_crag_path",
)
_PINNED_D4_BEIR_SUBSETS = ["scifact", "fiqa"]
_PINNED_D4_CRAG_SLICE_QUERIES = 500


def verify_dataset_manifest_dir(manifest_dir: Path) -> dict[str, Any]:
    resolved = manifest_dir.resolve()
    errors: list[dict[str, str]] = []
    checked: list[str] = []

    for bundle, filename in _EXPECTED_MANIFESTS.items():
        path = resolved / filename
        if not path.exists():
            errors.append({"path": str(path), "message": f"missing manifest for {bundle}"})
            continue

        payload = _read_yaml_mapping(path)
        checked.append(bundle)

        actual_bundle = str(payload.get("dataset_bundle", ""))
        if actual_bundle != bundle:
            errors.append(
                {
                    "path": str(path),
                    "message": f"dataset_bundle mismatch: expected {bundle!r}, got {actual_bundle!r}",
                }
            )

        for key in _REQUIRED_SIZE_KEYS:
            value = payload.get(key)
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                errors.append({"path": str(path), "message": f"{key} must be a positive integer"})
                continue
            if numeric <= 0:
                errors.append({"path": str(path), "message": f"{key} must be > 0"})

        version = payload.get("source_version")
        if not isinstance(version, str) or not version.strip():
            errors.append({"path": str(path), "message": "source_version must be a non-empty string"})

        checksum = payload.get("source_checksum_sha256")
        if not isinstance(checksum, str) or not _SHA256_RE.fullmatch(checksum.strip()):
            errors.append(
                {
                    "path": str(path),
                    "message": "source_checksum_sha256 must be a 64-character lowercase hex string",
                }
            )
        for key in _OPTIONAL_CACHE_CHECKSUM_KEYS:
            value = payload.get(key)
            if value in {None, ""}:
                continue
            text = str(value).strip()
            if not _SHA256_RE.fullmatch(text):
                errors.append(
                    {
                        "path": str(path),
                        "message": f"{key} must be a 64-character lowercase hex string when provided",
                    }
                )

        if bundle == "D4":
            _validate_d4_manifest(path=path, payload=payload, errors=errors)

    return {
        "pass": not errors,
        "error_count": len(errors),
        "manifest_dir": str(resolved),
        "checked_bundles": sorted(checked),
        "errors": errors,
    }


def _validate_d4_manifest(*, path: Path, payload: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    for key in ("crag_source", "crag_file", "crag_url"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append({"path": str(path), "message": f"D4 manifest missing non-empty {key}"})
    beir_subsets = payload.get("beir_subsets")
    if not isinstance(beir_subsets, list) or [str(item) for item in beir_subsets] != _PINNED_D4_BEIR_SUBSETS:
        errors.append(
            {
                "path": str(path),
                "message": f"D4 beir_subsets must equal {_PINNED_D4_BEIR_SUBSETS!r}",
            }
        )
    value = payload.get("crag_slice_queries")
    try:
        numeric_queries = int(value)
    except (TypeError, ValueError):
        errors.append({"path": str(path), "message": "D4 crag_slice_queries must be a positive integer"})
    else:
        if numeric_queries != _PINNED_D4_CRAG_SLICE_QUERIES:
            errors.append(
                {
                    "path": str(path),
                    "message": f"D4 crag_slice_queries must equal {_PINNED_D4_CRAG_SLICE_QUERIES}",
                }
            )

    value = payload.get("crag_slice_docs")
    try:
        numeric_docs = int(value)
    except (TypeError, ValueError):
        errors.append({"path": str(path), "message": "D4 crag_slice_docs must be a positive integer"})
    else:
        if numeric_docs <= 0:
            errors.append({"path": str(path), "message": "D4 crag_slice_docs must be > 0"})


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected YAML mapping at {path}")
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify dataset manifest coverage and pinned metadata")
    parser.add_argument("--manifest-dir", default="maxionbench/datasets/manifests")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_dataset_manifest_dir(Path(args.manifest_dir))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        state = "pass" if summary["pass"] else "fail"
        print(f"{state}: dataset manifest verification error_count={summary['error_count']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
