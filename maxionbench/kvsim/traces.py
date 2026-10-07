"""Load AgentX-format agent traces into compact sessions for the KV-cache simulator.

A trace row is one agent session: main-agent calls plus sub-agent groups, each call with its arrival
`t`, service time `api_time`, token counts, and the 64-token KV block ids of its prompt (`hash_ids`,
scoped to the session). Sub-agent groups become extra streams of the same session. Idle gaps longer
than `idle_cap_s` (a human away from the keyboard) are shortened to `idle_cap_s`, so a replay spends
its time on active work; the cap is reported with every result.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

BLOCK_TOKENS = 64
TRACE_FILE = "agentx/cc_traces_256k.jsonl"


@dataclass(frozen=True)
class Request:
    t: float  # arrival, seconds from session start (idle-capped)
    dur: float  # service time from the trace
    blocks: np.ndarray  # int32 prompt block ids, local to the session
    in_tokens: int
    out_tokens: int
    stream: int  # 0 = main agent, k = k-th sub-agent group
    next_t: float  # next arrival in the same stream, math.inf if last


@dataclass(frozen=True)
class Session:
    id: str
    requests: tuple[Request, ...]  # sorted by arrival
    span: float  # end of the last request


def _calls(row: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    group = 0
    for item in row["requests"]:
        if item.get("type") == "subagent":
            group += 1
            for inner in item["requests"]:
                yield group, inner
        else:
            yield 0, item


def build_session(row: dict[str, Any], idle_cap_s: float) -> Session:
    if row.get("block_size", BLOCK_TOKENS) != BLOCK_TOKENS:
        raise ValueError(f"session {row['id']}: block_size {row['block_size']} != {BLOCK_TOKENS}")
    calls = sorted(_calls(row), key=lambda c: c[1]["t"])
    shift, busy_until = 0.0, calls[0][1]["t"] if calls else 0.0
    timed = []
    for stream, c in calls:
        if c["t"] - busy_until > idle_cap_s:
            shift += c["t"] - busy_until - idle_cap_s
        busy_until = max(busy_until, c["t"] + c["api_time"])
        timed.append((stream, c["t"] - shift, c))
    first = timed[0][1] if timed else 0.0
    next_arrival: dict[int, float] = {}
    reqs: list[Request] = []
    for stream, t, c in reversed(timed):
        reqs.append(Request(
            t=t - first, dur=float(c["api_time"]), blocks=np.asarray(c["hash_ids"], dtype=np.int32),
            in_tokens=int(c["in"]), out_tokens=int(c["out"]), stream=stream,
            next_t=next_arrival.get(stream, math.inf)))
        next_arrival[stream] = t - first
    reqs.reverse()
    return Session(id=row["id"], requests=tuple(reqs), span=max((r.t + r.dur for r in reqs), default=0.0))


def load_sessions(path: Path, idle_cap_s: float = 300.0, limit: int | None = None) -> list[Session]:
    sessions = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(sessions) >= limit:
                break
            sessions.append(build_session(json.loads(line), idle_cap_s))
    return sessions
