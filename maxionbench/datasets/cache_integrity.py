"""Dataset cache integrity helpers (local file SHA-256 verification)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_sha256(value: Any, *, field_label: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_label} must be a 64-character lowercase hex sha256 string")
    return text


def sha256_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"expected file path but found non-file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_sha256(*, path: Path, expected_sha256: str, label: str) -> str:
    expected = normalize_sha256(expected_sha256, field_label=f"{label} expected_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def load_dataset_manifest(dataset_bundle: str) -> dict[str, Any] | None:
    root = Path(__file__).resolve().parent
    candidate = root / "manifests" / f"{str(dataset_bundle).lower()}.yaml"
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest must be a YAML mapping: {candidate}")
    return dict(payload)


def resolve_expected_sha256(
    *,
    config_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any] | None,
    config_key: str,
    manifest_key: str,
    label: str,
) -> str | None:
    expected, _ = resolve_expected_sha256_with_source(
        config_payload=config_payload,
        manifest_payload=manifest_payload,
        config_key=config_key,
        manifest_key=manifest_key,
        label=label,
    )
    return expected


def resolve_expected_sha256_with_source(
    *,
    config_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any] | None,
    config_key: str,
    manifest_key: str,
    label: str,
) -> tuple[str | None, str | None]:
    raw = config_payload.get(config_key)
    raw_source = f"config key {config_key}"
    if (raw is None or raw == "") and manifest_payload is not None:
        raw = manifest_payload.get(manifest_key)
        raw_source = f"manifest key {manifest_key}"
    if raw is None or raw == "":
        return None, None
    try:
        return normalize_sha256(raw, field_label=raw_source), raw_source
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
