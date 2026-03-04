"""Port allocation helpers for Slurm jobs."""

from __future__ import annotations

import os


def allocate_port(base: int = 20000, span: int = 20000, offset: int = 0) -> int:
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    if span <= 0:
        raise ValueError("span must be positive")
    return base + (job_id % span) + offset
