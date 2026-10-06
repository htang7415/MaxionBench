"""Provenance for result bundles: git state, host, tool versions, and secret redaction."""

from __future__ import annotations

import platform
import subprocess
from typing import Any, Callable, Mapping

from maxionbench.harness.results import RESULT_SCHEMA_VERSION, Provenance
from maxionbench.harness.secrets import MissingSecretError, load_gemini_key, redact
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.schemas.result_schema import stable_config_fingerprint, utc_now_iso


def git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def scrubber() -> tuple[Callable[[str], str], bool]:
    """Redact any configured API key from text bound for logs or result bundles; also report presence."""
    try:
        key = load_gemini_key()
    except MissingSecretError:
        return (lambda text: text), False
    return (lambda text: redact(text, key)), True


def make_provenance(spec: Mapping[str, Any], started_at: str, tools: Mapping[str, Any]) -> Provenance:
    return Provenance(
        git_commit=git(["rev-parse", "HEAD"]) or "unknown",
        # Untracked source counts: results from uncommitted code are not reproducible from git_commit.
        git_dirty=bool(git(["status", "--porcelain"])),
        spec_fingerprint=stable_config_fingerprint(dict(spec)),
        started_at=started_at,
        finished_at=utc_now_iso(),
        host=collect_system_info(),
        tools={"python": platform.python_version(), "harness_result_schema": RESULT_SCHEMA_VERSION, **tools},
    )
