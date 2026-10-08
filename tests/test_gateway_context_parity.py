"""The Go gateway's context manager (gateway/internal/ctxmgr) must trim exactly like agents/context.py.
This test pins the Python behavior in a fixture the Go tests replay; set MAXIONBENCH_REGEN_PARITY=1 to
rewrite it after an intended change to the policies."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from maxionbench.agents.context import MASK_TEXT, make_policy

FIXTURE = Path(__file__).resolve().parents[1] / "gateway/internal/ctxmgr/testdata/parity.json"
CASES = [("mask+cache", 2, 300, 0), ("mask+cache", 4, 900, 0), ("window+cache", 3, 300, 0), ("window+cache", 8, 900, 0),
         ("mask+cache", 1, 150, 200), ("window+cache", 3, 200, 150)]
WORDS = ("cache", "kv", "prefix", "tool", "résumé", "naïve", "数据", "agent", "page", "search")


def agent_history(seed: int) -> tuple[list[dict[str, Any]], list[int]]:
    """A history and the lengths the client sent it at (one call per step)."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    steps = [2]
    for e in range(rng.randint(4, 14)):
        if rng.random() < 0.2:
            out.append({"role": "assistant", "content": " ".join(rng.choices(WORDS, k=rng.randint(1, 20)))})
            out.append({"role": "user", "content": "go on"})
        else:
            ids = [f"c{e}-{i}" for i in range(rng.randint(1, 2))]
            out.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": i, "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": i})}}
                for i in ids]})
            out += [{"role": "tool", "tool_call_id": i,
                     "content": " ".join(rng.choices(WORDS, k=rng.choice((5, 60, 250))))} for i in ids]
        steps.append(len(out))
    return out, steps


def encode(view: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[Any]:
    """Each view message as its history index, or ["mask", index] for a masked tool result."""
    out: list[Any] = []
    for m in view:
        idx = next(i for i, h in enumerate(history) if h.get("tool_call_id", h.get("content")) ==
                   m.get("tool_call_id", m.get("content")) and h["role"] == m["role"]
                   and (h is m or m["content"] == MASK_TEXT))
        out.append(idx if history[idx] is m else ["mask", idx])
    return out


def build() -> dict[str, Any]:
    histories, cases = [], []
    for seed in range(6):
        history, steps = agent_history(seed)
        histories.append({"history": history, "steps": steps})
        for name, keep, budget, growth in CASES:
            policy = make_policy(name, keep=keep, budget_tokens=budget, min_growth=growth)
            views = [encode(policy.view(history[:n]), history) for n in steps]
            cases.append({"history": seed, "policy": name, "keep": keep, "budget_tokens": budget, "min_growth": growth,
                          "views": views})
    return {"mask_text": MASK_TEXT, "histories": histories, "cases": cases}


def test_parity_fixture_matches_python() -> None:
    data = build()
    if os.environ.get("MAXIONBENCH_REGEN_PARITY") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == data


def test_fixture_covers_appends_and_edits() -> None:
    """The Go replay must see both branches: views that extend the last one and views re-rendered."""
    pairs = [(a, b) for case in build()["cases"] for a, b in zip(case["views"], case["views"][1:])]
    edits = sum(1 for a, b in pairs if b[:len(a)] != a)
    assert 0.2 * len(pairs) < edits < 0.8 * len(pairs)
