from __future__ import annotations

import json
import random
from typing import Any

import pytest

from maxionbench.agents.context import (
    MASK_TEXT, SUMMARY_PREFIX, CacheAware, Mask, Window, estimate_tokens, make_policy, split,
)

POLICIES = ("full", "truncate", "window", "mask", "summarize", "truncate+cache", "window+cache", "mask+cache")
SEEDS = range(25)


def summarizer(messages: Any) -> str:
    return f"{len(messages)} messages condensed"


def policy(name: str) -> Any:
    params = {"summarize": {"trigger_tokens": 12_000}, "window": {"keep": 3}, "mask": {"keep": 2},
              "truncate": {"max_tokens": 500}}.get(name.split("+")[0], {})
    if name.endswith("+cache"):
        params = {**params, "budget_tokens": 12_000}
    return make_policy(name, summarizer if name == "summarize" else None, **params)


def history(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for e in range(rng.randint(1, 30)):
        ids = [f"c{e}-{i}" for i in range(rng.randint(1, 3))]
        out.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": i})}}
            for i in ids]})
        out += [{"role": "tool", "tool_call_id": i, "content": "x" * rng.choice((50, 800, 4_000, 20_000))}
                for i in ids]
    return out


def steps(h: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The history as the agent sees it after each exchange."""
    head, exchanges = split(h)
    return [head + [m for ex in exchanges[:k] for m in ex] for k in range(1, len(exchanges) + 1)]


def run(name: str, seed: int) -> tuple[Any, list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    p, hs = policy(name), steps(history(seed))
    return p, hs, [p.view(h) for h in hs]


def is_prefix(a: list[Any], b: list[Any]) -> bool:
    return b[:len(a)] == a


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_views_keep_task_newest_exchange_and_call_result_pairing(name: str, seed: int) -> None:
    _, hs, views = run(name, seed)
    for h, v in zip(hs, views):
        assert v[:2] == h[:2]  # system prompt and task are never dropped
        newest = {m["tool_call_id"] for m in split(h)[1][-1] if m["role"] == "tool"}
        assert newest <= {m.get("tool_call_id") for m in v}  # the results just returned are always visible
        called: list[str] = []
        for m in v:
            if m["role"] == "assistant":
                called += [c["id"] for c in m.get("tool_calls") or ()]
            if m["role"] == "tool":
                assert m["tool_call_id"] in called  # no orphan result
        assert set(called) == {m["tool_call_id"] for m in v if m["role"] == "tool"}  # no call without its result


@pytest.mark.parametrize("name", POLICIES)
def test_policies_are_deterministic(name: str) -> None:
    for seed in SEEDS:
        assert run(name, seed)[2] == run(name, seed)[2]


def test_full_is_identity_and_append_only_policies_only_append() -> None:
    for seed in SEEDS:
        _, hs, views = run("full", seed)
        assert views == hs
        for name in ("truncate", "truncate+cache"):
            views = run(name, seed)[2]
            assert all(is_prefix(a, b) for a, b in zip(views, views[1:]))


@pytest.mark.parametrize("seed", SEEDS)
def test_size_rules_of_each_policy(seed: int) -> None:
    for v in run("truncate", seed)[2]:
        assert all(len(m["content"]) <= 4 * 500 + 40 for m in v if m["role"] == "tool")
    for v in run("window", seed)[2]:
        assert len(split(v)[1]) <= 3
    for v in run("mask", seed)[2]:
        exchanges = split(v)[1]
        assert all(m["content"] == MASK_TEXT for ex in exchanges[:-2] for m in ex if m["role"] == "tool")
        assert all(m["content"] != MASK_TEXT for ex in exchanges[-2:] for m in ex if m["role"] == "tool")
    p, hs, views = run("summarize", seed)
    for h, v in zip(hs, views):
        summarized = [m for m in v if str(m.get("content") or "").startswith(SUMMARY_PREFIX)]
        assert len(summarized) <= 1 and (estimate_tokens(v) <= 12_000 or len(split(v)[1]) <= 3)
    assert p.compactions == sum(1 for a, b in zip(views, views[1:]) if not is_prefix(a, b))


@pytest.mark.parametrize("seed", SEEDS)
def test_cache_aware_views_only_append_between_edits_and_respect_budget(seed: int) -> None:
    for base in ("window", "mask"):
        p, hs, views = policy(f"{base}+cache"), steps(history(seed)), []
        edits = []
        for h in hs:
            views.append(p.view(h))
            edits.append(p.edits)
            base_view = policy(base).view(h)
            assert estimate_tokens(views[-1]) <= 12_000 or views[-1] == base_view
        for k in range(1, len(views)):
            if edits[k] == edits[k - 1]:
                assert is_prefix(views[k - 1], views[k])
            else:
                assert views[k] == policy(base).view(hs[k])  # an edit re-renders with the base policy


def test_cache_aware_keeps_more_prefixes_than_its_base() -> None:
    def reused(views: list[Any]) -> int:
        return sum(is_prefix(a, b) for a, b in zip(views, views[1:]))

    for seed in SEEDS:
        for base in ("window", "mask"):
            assert reused(run(f"{base}+cache", seed)[2]) >= reused(run(base, seed)[2])
    long_history = steps(history(3))
    window = Window(keep=3)
    cached = CacheAware(Window(keep=3), budget_tokens=12_000)
    plain_views = [window.view(h) for h in long_history]
    cached_views = [cached.view(h) for h in long_history]
    assert reused(cached_views) > reused(plain_views)


def test_make_policy_rejects_unknown_and_stateful_wrapping() -> None:
    with pytest.raises(KeyError):
        make_policy("lru")
    with pytest.raises(ValueError):
        make_policy("summarize+cache", summarizer)
    with pytest.raises(ValueError):
        make_policy("summarize")
    assert make_policy("mask+cache", keep=1).name == "mask1+cache64k"
    assert isinstance(make_policy("mask", keep=1), Mask)


def test_policies_honour_token_counts_of_metadata_only_messages() -> None:
    def msg(role: str, tokens: int, key: str) -> dict[str, Any]:
        return {"role": role, "content": key, "tokens": tokens}

    h = [msg("system", 3_000, "s"), msg("user", 100, "u")]
    for e in range(4):
        h += [msg("assistant", 30, f"a{e}"), msg("tool", 10_000, f"t{e}")]
    assert estimate_tokens(h) == 3_100 + 4 * 10_030
    truncated = make_policy("truncate", max_tokens=2_000).view(h)
    assert all(m["tokens"] == 2_008 and m["content"].endswith("[truncated]") for m in truncated if m["role"] == "tool")
    masked = make_policy("mask", keep=1).view(h)
    assert [m["content"] == MASK_TEXT and "tokens" not in m for m in masked if m["role"] == "tool"] == [True] * 3 + [False]
    assert estimate_tokens(masked) < 3_100 + 4 * 30 + 10_000 + 3 * 30
    summarized = make_policy("summarize", summarizer, trigger_tokens=20_000, keep=1).view(h)
    assert estimate_tokens(summarized) <= 20_000 and summarized[2]["content"].startswith(SUMMARY_PREFIX)
