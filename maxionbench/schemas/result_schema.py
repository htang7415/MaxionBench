"""Timestamp and config-fingerprint helpers for result provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


def utc_now_iso() -> str:
    """Return UTC timestamp in stable ISO-8601 format."""

    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def stable_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash resolved config deterministically for reproducibility tracking."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
