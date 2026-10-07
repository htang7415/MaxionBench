"""GitHub Copilot coding-agent traces 2026 (Azure Public Dataset): sessions streamed from the daily archives.

Each archive `copilot/date.<day>.tar.gz` holds gzipped JSONL shards, one session per line (turns of LLM
calls with per-segment token counts, and tool batches). Metadata only: no prompts, code, or tool output.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tarfile
from typing import Any, Iterator

from maxionbench.datasets.sources import DATASET_ROOT, verified_path

DAYS = tuple(f"2026-06-0{d}" for d in range(1, 8))


def archive(day: str) -> str:
    return f"copilot/date.{day}.tar.gz"


def iter_sessions(day: str, root: Path = DATASET_ROOT, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Sessions of one day in archive order (shards as stored), without extracting to disk."""
    n = 0
    with tarfile.open(verified_path(archive(day), root), "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".jsonl.gz"):
                continue
            with gzip.open(tar.extractfile(member)) as fh:  # type: ignore[arg-type]
                for line in fh:
                    yield json.loads(line)
                    n += 1
                    if limit is not None and n >= limit:
                        return
