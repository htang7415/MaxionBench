"""Copilot sessions as seen through a context policy, written as AgentX-format traces for the KV simulator.

A Copilot call's metadata lists its prompt's messages in order (type, role, token count) without text.
Each message becomes a placeholder keyed by (sequenceId, type, role) with the token count it had when it
first appeared in the session, so the same message is the same content, of the same length, in every call. A context policy (maxionbench.agents.context) turns each
call's history into the prompt actually sent; the prompt is cut into 64-token blocks whose ids are
chained (a block's id depends on its content and every block before it), so two prompts share block
ids exactly as far as they share a token prefix, which is what vLLM's prefix cache reuses.

Copilot sometimes rewrites its own history between calls (its compaction); the policy's state is reset
there. `summarize` adds the summarizer's own request before the call (prompt: the summarized messages
plus an instruction appended at the end, so it reuses the cached prefix; output SUMMARY_TOKENS).

These are counterfactual replays: the agent's trajectory is Copilot's, under each policy's prompts. In
Step 3 policies also changed what the agent did (more or fewer calls); that is not modeled here.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Iterator

from maxionbench.agents.context import Message, estimate_tokens, make_policy
from maxionbench.datasets.loaders.copilot import iter_archive
from maxionbench.kvsim.traces import BLOCK_TOKENS

SUMMARY_TOKENS = 1_000
SUMMARY_S = 3.0  # summarizer service time; the call itself starts this much later
SUMMARIZE_INSTRUCTION_TOKENS = 60


def history(call: dict[str, Any]) -> list[Message]:
    roles = {"system", "user", "assistant", "tool"}
    return [{"role": m["role"] if m["role"] in roles else "user", "content": f"{m['sequenceId']}|{m['type']}|{m['role']}",
             "tokens": int(m["token_len"] or 0)}
            for m in sorted(call["message_metadata"] or [], key=lambda m: m["sequenceId"])]


class BlockIds:
    """Prefix-chained 64-token block ids, interned per session (deterministic, collision-free)."""

    def __init__(self) -> None:
        self._ids: dict[tuple[int, tuple[tuple[str, int, int], ...]], int] = {}

    def __call__(self, prompt: list[Message]) -> list[int]:
        out: list[int] = []
        prev, pieces, filled = -1, [], 0
        for m in prompt:
            key, n, pos = str(m["content"]), estimate_tokens([m]), 0
            while pos < n:
                take = min(BLOCK_TOKENS - filled, n - pos)
                pieces.append((key, pos, pos + take))
                pos, filled = pos + take, filled + take
                if filled == BLOCK_TOKENS:
                    prev = self._ids.setdefault((prev, tuple(pieces)), len(self._ids))
                    out.append(prev)
                    pieces, filled = [], 0
        if pieces:  # the partial last block
            out.append(self._ids.setdefault((prev, tuple(pieces)), len(self._ids)))
        return out


def _end_s(ts: str) -> float:
    return datetime.fromisoformat(ts[:26].rstrip("Z") + "+00:00").timestamp()


def calls_of(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Calls with token accounting and prompt metadata, in time order (`timestamp` marks a call's end)."""
    calls = [c for t in session["turns"] or [] for c in t["llm_calls"] or []
             if (c["tokens"] or {}).get("prompt") and c["message_metadata"]]
    return sorted(calls, key=lambda c: _end_s(c["timestamp"]))


def session_trace(session: dict[str, Any], policy: str, params: dict[str, Any]) -> dict[str, Any]:
    pending: list[list[Message]] = []
    count = [0]

    def summarizer(messages: Any) -> str:
        pending.append(list(messages) + [{"role": "user", "content": "summarize", "tokens": SUMMARIZE_INSTRUCTION_TOKENS}])
        count[0] += 1
        return f"summary {count[0]} ".ljust(4 * SUMMARY_TOKENS, ".")

    view_of = make_policy(policy, summarizer if policy.startswith("summarize") else None, **params)
    blocks = BlockIds()
    first_len: dict[str, int] = {}
    requests, prev_keys = [], None
    for c in calls_of(session):
        # Copilot's per-message token counts drift between calls for unchanged messages (they appear to be the
        # prompt total apportioned by length), so a message keeps the count it had when it first appeared
        h = [{**m, "tokens": first_len.setdefault(m["content"], m["tokens"])} for m in history(c)]
        keys = [m["content"] for m in h]
        if prev_keys is not None and keys[:len(prev_keys)] != prev_keys:
            view_of.reset()  # Copilot rewrote its history: the policy starts over from this call
        prev_keys = keys
        prompt = view_of.view(h)
        dur = float(c["duration_ms"] or 0) / 1000
        start = _end_s(c["timestamp"]) - dur
        for sp in pending:
            requests.append({"t": round(start, 3), "api_time": SUMMARY_S, "in": estimate_tokens(sp),
                             "out": SUMMARY_TOKENS, "hash_ids": blocks(sp)})
            start += SUMMARY_S
        pending.clear()
        requests.append({"t": round(start, 3), "api_time": round(dur, 3), "in": estimate_tokens(prompt),
                         "out": int(c["tokens"].get("completion") or 0), "hash_ids": blocks(prompt)})
    return {"id": session["session_id"], "block_size": BLOCK_TOKENS, "policy": policy, "requests": requests}


def sample_ids(archive: Path, sessions: int, seed: int) -> set[str]:
    eligible = sorted(s["session_id"] for s in iter_archive(archive) if len(calls_of(s)) >= 2)
    return set(random.Random(seed).sample(eligible, min(sessions, len(eligible))))


def iter_policy_traces(archive: Path, ids: set[str], policy: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for s in iter_archive(archive):
        if s["session_id"] in ids:
            yield session_trace(s, policy, params)


def derive_policy_trace(src: Path, dest: Path, *, sessions: int, seed: int, policy: str,
                        policy_params: dict[str, Any] | None = None) -> None:
    """Manifest deriver: `sessions` seeded sessions of one Copilot day archive under `policy`."""
    ids = sample_ids(src, sessions, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in iter_policy_traces(src, ids, policy, dict(policy_params or {})):
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(dest)
