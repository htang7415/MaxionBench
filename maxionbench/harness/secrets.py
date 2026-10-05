"""API key loading that resists accidental disclosure.

Keys are read at runtime only (env first, then a local file that is git- and docker-ignored), are
wrapped so repr/str/JSON/pickle never reveal them, and can be redacted from any text before it is
logged or written to a result bundle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

GEMINI_ENV = "GEMINI_API_KEY"
GEMINI_FILE_ENV = "MAXIONBENCH_GEMINI_KEY_FILE"
DEFAULT_GEMINI_KEY_FILE = Path("docs/gemini_api.txt")
REDACTED = "[REDACTED]"


class MissingSecretError(RuntimeError):
    pass


class Secret:
    """Opaque holder; call `reveal()` only at the point of use (e.g. an HTTP header)."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value:
            raise ValueError("secret must be non-empty")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __reduce__(self) -> object:
        raise TypeError("Secret objects must not be pickled")


def load_gemini_key(env: Mapping[str, str] | None = None) -> Secret:
    env = os.environ if env is None else env
    value = (env.get(GEMINI_ENV) or "").strip()
    if value:
        return Secret(value)
    path = Path(env.get(GEMINI_FILE_ENV) or DEFAULT_GEMINI_KEY_FILE).expanduser()
    if not path.is_file():
        raise MissingSecretError(f"set {GEMINI_ENV} or create {path} (git- and docker-ignored)")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise MissingSecretError(f"{path} is empty")
    return Secret(value)


def gemini_key_present(env: Mapping[str, str] | None = None) -> bool:
    """Provenance-safe: reports presence only, never the value."""
    try:
        load_gemini_key(env)
    except MissingSecretError:
        return False
    return True


def redact(text: str, *secrets: Secret) -> str:
    for secret in secrets:
        text = text.replace(secret.reveal(), REDACTED)
    return text
