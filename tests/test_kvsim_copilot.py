from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from maxionbench.agents.context import MASK_TEXT, estimate_tokens
from maxionbench.kvsim.copilot import SUMMARY_TOKENS, BlockIds, session_trace
from maxionbench.kvsim.traces import build_session


def msg(key: str, tokens: int, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": key, "tokens": tokens}


def test_block_ids_are_shared_exactly_as_far_as_the_token_prefix() -> None:
    ids = BlockIds()
    a = ids([msg("sys", 100), msg("u", 50)])  # 150 tokens: blocks 0-63, 64-127, partial 128-149
    assert len(a) == 3 and ids([msg("sys", 100), msg("u", 50)]) == a
    longer = ids([msg("sys", 100), msg("u", 50), msg("t", 200, "tool")])
    assert longer[:2] == a[:2] and longer[2] != a[2]  # the partial block is completed differently
    changed = ids([msg("sys", 100), msg("x", 50), msg("t", 200, "tool")])
    assert changed[0] == longer[0] and changed[1:] != longer[1:]  # block 1 holds tokens of the changed message
    assert len(set(changed[1:]) & set(longer[1:])) == 0  # chained: nothing after a change is shared


def call(end: str, prompt: int, segments: list[tuple[int, str, str, int]], dur_ms: float = 2_000.0) -> dict[str, Any]:
    return {"timestamp": f"2026-06-01T{end}.000000000Z", "duration_ms": dur_ms,
            "tokens": {"prompt": prompt, "completion": 20},
            "message_metadata": [{"sequenceId": s, "type": t, "role": r, "token_len": n} for s, t, r, n in segments]}


def session() -> dict[str, Any]:
    base = [(0, "System", "system", 3_000), (1, "UserMessage", "user", 100)]
    exchanges = [(2 + 2 * e, "FunctionCalls", "assistant", 40) for e in range(4)]
    results = [(3 + 2 * e, "FunctionCalls", "tool", 9_000) for e in range(4)]
    segs = [base + [x for e in range(k) for x in (exchanges[e], results[e])] for k in range(1, 5)]
    calls = [call(f"10:00:{10 * (k + 1):02d}", sum(s[3] for s in segs[k]), segs[k]) for k in range(4)]
    rewritten = [(0, "System", "system", 3_000), (1, "UserMessage", "user", 100), (20, "History", "user", 500)]
    calls.append(call("10:01:00", 3_600, rewritten))  # Copilot compacted its own history
    return {"session_id": "s1", "turns": [{"llm_calls": calls, "tool_batches": []}]}


def test_full_policy_prompts_extend_each_other_and_load_as_kvsim_sessions() -> None:
    row = session_trace(session(), "full", {})
    reqs = row["requests"]
    assert [r["in"] for r in reqs] == [12_140, 21_180, 30_220, 39_260, 3_600]
    for a, b in zip(reqs[:3], reqs[1:4]):
        full_blocks = a["in"] // 64
        assert b["hash_ids"][:full_blocks] == a["hash_ids"][:full_blocks]  # each prompt extends the last
    end = datetime(2026, 6, 1, 10, 0, 10, tzinfo=timezone.utc).timestamp()
    assert reqs[0]["t"] == end - 2.0 and reqs[0]["api_time"] == 2.0  # `timestamp` marks the call's end
    s = build_session(row, idle_cap_s=300.0)
    assert len(s.requests) == 5 and s.requests[0].t == 0.0


def test_mask_breaks_the_shared_prefix_at_the_first_masked_result() -> None:
    full, masked = session_trace(session(), "full", {}), session_trace(session(), "mask", {"keep": 1})
    third = masked["requests"][2]
    assert third["in"] < full["requests"][2]["in"]
    shared = next(i for i, (x, y) in enumerate(zip(masked["requests"][1]["hash_ids"], third["hash_ids"])) if x != y)
    mask_tokens = estimate_tokens([{"role": "tool", "content": MASK_TEXT}])
    # call 2 masks exchange 0 (already masked in call 1) and now exchange 1: the prefix breaks at exchange 1's result
    assert shared == (3_000 + 100 + 40 + mask_tokens + 40) // 64


def test_summarize_adds_its_own_request_and_policy_state_resets_on_rewrite() -> None:
    row = session_trace(session(), "summarize", {"trigger_tokens": 25_000, "keep": 1})
    summaries = [r for r in row["requests"] if r["out"] == SUMMARY_TOKENS]
    assert len(summaries) >= 1
    first = row["requests"].index(summaries[0])
    nxt = row["requests"][first + 1]
    assert nxt["t"] == summaries[0]["t"] + 3.0 and nxt["in"] <= 25_000
    assert row["requests"][-1]["in"] == 3_600  # after Copilot's rewrite the policy starts over: nothing to cut
    cached = session_trace(session(), "mask+cache", {"keep": 1, "budget_tokens": 25_000})
    assert cached["requests"][-1]["in"] == 3_600


def test_unchanged_messages_keep_their_first_token_count_so_prefixes_stay_shared() -> None:
    s = session()
    for k, c in enumerate(s["turns"][0]["llm_calls"][:4]):
        c["message_metadata"][0]["token_len"] = 3_000 - 50 * k  # Copilot's drifting count for the same system prompt
    reqs = session_trace(s, "full", {})["requests"]
    for a, b in zip(reqs[:3], reqs[1:4]):
        assert b["hash_ids"][:a["in"] // 64] == a["hash_ids"][:a["in"] // 64]
