"""Healthcheck polling helpers."""

from __future__ import annotations

import time
from typing import Callable


def wait_for_healthy(
    health_fn: Callable[[], bool],
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health_fn():
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"healthcheck did not pass within {timeout_s}s")
