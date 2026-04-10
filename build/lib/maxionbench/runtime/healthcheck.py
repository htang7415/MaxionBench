"""Healthcheck polling helpers."""

from __future__ import annotations

import socket
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


def wait_for_port_ready(
    host: str,
    port: int,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
    connect_timeout_s: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=float(connect_timeout_s)):
                return
        except (socket.timeout, OSError):
            time.sleep(poll_interval_s)
    raise TimeoutError(f"port {host}:{port} did not become reachable within {timeout_s}s")
