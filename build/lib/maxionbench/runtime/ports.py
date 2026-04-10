"""Port allocation helpers for Slurm jobs."""

from __future__ import annotations

import os
from typing import Sequence


def allocate_port(base: int = 20000, span: int = 20000, offset: int = 0) -> int:
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if span <= 0:
        raise ValueError("span must be positive")
    return base + (((job_id * 100) + task_id) % span) + offset


def allocate_port_range(
    *,
    count: int,
    base: int = 20000,
    span: int = 20000,
    offset: int = 0,
) -> list[int]:
    if count < 1:
        raise ValueError("count must be >= 1")
    start = allocate_port(base=base, span=span, offset=offset)
    ports = [start + idx for idx in range(count)]
    if ports[-1] >= base + span:
        raise ValueError("requested port range exceeds span")
    return ports


def allocate_named_ports(
    names: Sequence[str],
    *,
    base: int = 20000,
    span: int = 20000,
    offset: int = 0,
) -> dict[str, int]:
    unique = []
    seen = set()
    for name in names:
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    ports = allocate_port_range(count=len(unique), base=base, span=span, offset=offset)
    return {name: port for name, port in zip(unique, ports)}
