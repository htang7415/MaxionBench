"""Helpers for companion provenance artifacts written with conformance matrices."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any


def conformance_provenance_path(matrix_path: Path) -> Path:
    resolved = matrix_path.expanduser()
    stem = resolved.stem if resolved.suffix else resolved.name
    return resolved.with_name(f"{stem}.provenance.json")


def build_conformance_provenance(*, config_dir: Path, matrix_path: Path) -> dict[str, Any]:
    container_image = str(os.environ.get("MAXIONBENCH_CONTAINER_IMAGE", "")).strip()
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_dir": _public_path(config_dir),
        "matrix_path": _public_path(matrix_path),
        "python_executable": "redacted",
        "container_runtime": str(os.environ.get("MAXIONBENCH_CONTAINER_RUNTIME", "")).strip().lower(),
        "container_image": _public_container_image(container_image),
        "hostname": "redacted",
    }


def _public_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _public_container_image(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith(("/", "~", ".")):
        resolved = str(Path(value).expanduser().resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
        return f"local-path-sha256:{digest}"
    return value
