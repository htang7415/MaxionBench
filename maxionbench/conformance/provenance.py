"""Helpers for companion provenance artifacts written with conformance matrices."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import sys
from typing import Any


def conformance_provenance_path(matrix_path: Path) -> Path:
    resolved = matrix_path.expanduser()
    stem = resolved.stem if resolved.suffix else resolved.name
    return resolved.with_name(f"{stem}.provenance.json")


def build_conformance_provenance(*, config_dir: Path, matrix_path: Path) -> dict[str, Any]:
    container_image = str(os.environ.get("MAXIONBENCH_CONTAINER_IMAGE", "")).strip()
    resolved_container_image = ""
    if container_image:
        resolved_container_image = str(Path(container_image).expanduser().resolve())
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_dir": str(config_dir.expanduser().resolve()),
        "matrix_path": str(matrix_path.expanduser().resolve()),
        "python_executable": sys.executable,
        "container_runtime": str(os.environ.get("MAXIONBENCH_CONTAINER_RUNTIME", "")).strip().lower(),
        "container_image": resolved_container_image,
        "hostname": socket.gethostname(),
    }
