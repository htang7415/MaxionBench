import json
import math

import numpy as np
import pytest

from maxionbench.kvsim.sim import FreeBlocks, Replica, SimParams, simulate
from maxionbench.kvsim.traces import Request, Session, build_session


def _req(t, dur, blocks, stream=0, next_t=math.inf, out=64):
    return Request(t=t, dur=dur, blocks=np.asarray(blocks, dtype=np.int32), in_tokens=64 * len(blocks),
                   out_tokens=out, stream=stream, next_t=next_t)


def _session(sid, reqs):
    return Session(id=sid, requests=tuple(reqs), span=max(r.t + r.dur for r in reqs))


def test_build_session_flattens_subagents_caps_idle_and_links_streams():
    row = {"id": "s", "block_size": 64, "requests": [
        {"t": 10.0, "in": 128, "out": 5, "hash_ids": [0, 1], "api_time": 1.0, "type": "s"},
        {"t": 12.0, "type": "subagent", "requests": [
            {"t": 12.0, "in": 64, "out": 5, "hash_ids": [0], "api_time": 1.0, "type": "n"},
            {"t": 14.0, "in": 128, "out": 5, "hash_ids": [0, 2], "api_time": 1.0, "type": "n"}]},
        {"t": 1015.0, "in": 192, "out": 5, "hash_ids": [0, 1, 3], "api_time": 1.0, "type": "s"},
    ]}
    s = build_session(row, idle_cap_s=100.0)
    assert [r.t for r in s.requests] == [0.0, 2.0, 4.0, 105.0]  # 1000 s idle gap capped to 100 s
    assert [r.stream for r in s.requests] == [0, 1, 1, 0]
    assert s.requests[0].next_t == 105.0 and s.requests[1].next_t == 4.0 and math.isinf(s.requests[2].next_t)
    assert s.span == 106.0


def test_free_blocks_without_protection_is_lru():
    f = FreeBlocks()
    for k in (1, 2, 3):
        f.release(k, now=0.0, window=0.0, expected=0.0)
    f.take(2)
    assert [f.evict(1.0), f.evict(1.0), f.evict(1.0)] == [1, 3, None]


def test_free_blocks_prefers_lapsed_then_latest_return():
    f = FreeBlocks()
    f.release(1, now=0.0, window=100.0, expected=50.0)
    f.release(2, now=0.0, window=100.0, expected=90.0)
    f.release(3, now=0.0, window=5.0, expected=5.0)
    assert f.evict(10.0) == 3  # protection lapsed
    assert f.evict(10.0) == 2  # latest expected return among the protected
    f.take(1)
    assert f.evict(10.0) is None and len(f) == 0


def test_replica_cached_prefix_stops_at_first_gap():
    r = Replica(capacity_blocks=10)
    r.ref = {1: 0, 2: 0, 4: 0}
    assert r.cached_prefix([1, 2, 3, 4]) == 2


def _two_agents():
    # Agent A pauses 60 s between calls; agent B makes an unrelated burst during the pause that does
    # not fit beside A's context, so LRU evicts A while pause-aware retention keeps it.
    a = _session("a", [_req(0, 1, range(0, 8), next_t=61), _req(61, 1, range(0, 10))])
    b = _session("b", [_req(5, 1, range(100, 106), next_t=7), _req(7, 1, range(200, 206), next_t=9),
                       _req(9, 1, range(300, 306))])
    # Later sessions keep the run going past A's return (measurement stops when the last one starts).
    filler = [_session(f"f{i}", [_req(100, 1, [1000 + i])]) for i in range(3)]
    return [a, b, *filler]


def test_oracle_retention_keeps_paused_agent_that_lru_evicts():
    sessions = _two_agents()
    common = dict(replicas=1, concurrency=2, capacity_tokens=64 * 16, routing="session", warmup_s=0.0)
    # Pick a seed whose shuffle starts sessions a and b together.
    seed = next(s for s in range(100) if _first_two(s, len(sessions)) == {0, 1})
    lru = simulate(sessions, SimParams(**common, eviction="lru"), seed)
    oracle = simulate(sessions, SimParams(**common, eviction="oracle"), seed)
    assert oracle["miss_evicted_share"] < lru["miss_evicted_share"]
    assert oracle["token_hit_rate"] > lru["token_hit_rate"]
    assert lru["overflow_rate"] == oracle["overflow_rate"] == 0.0


def _first_two(seed, n):
    import random
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return set(order[:2])


def test_round_robin_misses_are_attributed_to_routing():
    s = [_session(f"s{i}", [_req(0, 1, [0, 1, 2], next_t=2), _req(2, 1, [0, 1, 2, 3])]) for i in range(4)]
    m = simulate(s, SimParams(replicas=2, concurrency=1, capacity_tokens=64 * 64, routing="round_robin",
                              warmup_s=0.0), seed=0)
    assert m["miss_routing_share"] > 0
    sticky = simulate(s, SimParams(replicas=2, concurrency=1, capacity_tokens=64 * 64, routing="session",
                                   warmup_s=0.0), seed=0)
    assert sticky["miss_routing_share"] == 0 and sticky["token_hit_rate"] > m["token_hit_rate"]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SimParams(replicas=1, concurrency=1, capacity_tokens=1 << 20, eviction="fifo")


def test_kvsim_spec_plan(tmp_path):
    from maxionbench.kvsim.__main__ import load_spec, plan
    spec = tmp_path / "k.yaml"
    spec.write_text(json.dumps({"schema_version": "maxionbench-kvsim-v1", "name": "k", "seed": 2, "repeats": 2,
                                "params": {"replicas": 2}, "matrix": {"eviction": ["lru", "oracle"]}}))
    trials = plan(load_spec(spec))
    assert [(c, r, s) for c, _, r, s in trials] == [
        ("eviction=lru", 0, 2000), ("eviction=lru", 1, 2001), ("eviction=oracle", 0, 2000),
        ("eviction=oracle", 1, 2001)]
    bad = tmp_path / "bad.yaml"
    bad.write_text(json.dumps({"schema_version": "maxionbench-kvsim-v1", "name": "b", "matrix": {"nope": [1]}}))
    with pytest.raises(ValueError):
        load_spec(bad)


def test_predicted_without_error_matches_oracle():
    sessions = _two_agents()
    common = dict(replicas=1, concurrency=2, capacity_tokens=64 * 16, routing="session", warmup_s=0.0)
    oracle = simulate(sessions, SimParams(**common, eviction="oracle"), 3)
    assert simulate(sessions, SimParams(**common, eviction="predicted", noise_sigma=0.0), 3) == oracle


def test_cpu_tier_turns_evictions_into_loads():
    sessions = _two_agents()
    common = dict(replicas=1, concurrency=2, capacity_tokens=64 * 16, routing="session", warmup_s=0.0,
                  eviction="lru")
    seed = next(s for s in range(100) if _first_two(s, len(sessions)) == {0, 1})
    gpu_only = simulate(sessions, SimParams(**common), seed)
    tiered = simulate(sessions, SimParams(**common, cpu_capacity_tokens=64 * 64, load_cost=0.25), seed)
    assert gpu_only["cpu_loaded_share"] == 0 and gpu_only["miss_evicted_share"] > 0
    assert tiered["cpu_loaded_share"] > 0 and tiered["miss_evicted_share"] == 0
    assert tiered["token_hit_rate"] == gpu_only["token_hit_rate"]  # GPU hits unchanged; loads replace recompute
    assert tiered["recomputed_tokens_per_request"] < gpu_only["recomputed_tokens_per_request"]
    loaded_per_request = tiered["cpu_loaded_share"] * tiered["prompt_tokens_per_request"]
    assert tiered["prefill_cost_tokens_per_request"] == pytest.approx(
        tiered["recomputed_tokens_per_request"] + 0.25 * loaded_per_request)


def test_tiered_routing_follows_kv_demoted_to_cpu():
    r0 = Replica(capacity_blocks=4, cpu_blocks=8)
    r0.cpu.update({1: None, 2: None, 3: None})
    assert r0.cached_prefix([1, 2, 3]) == 0 and r0.cpu_run([1, 2, 3], 0) == 3
    # A runs on replica 0 and pauses; B's long call on replica 0 pushes A's context to its CPU tier.
    # When A returns, GPU-only routing prefers idle replica 1 and recomputes; tier-aware routing
    # goes back to replica 0 and loads A's prefix from CPU.
    a = _session("a", [_req(0, 1, range(0, 8), next_t=30), _req(30, 1, range(0, 9))])
    b = _session("b", [_req(2, 100, range(100, 108))])
    filler = [_session(f"f{i}", [_req(200, 1, [1000 + i])]) for i in range(3)]
    sessions = [a, b, *filler]
    seed = next(s for s in range(100) if _first_two(s, len(sessions)) == {0, 1})
    common = dict(replicas=2, concurrency=2, capacity_tokens=64 * 10, cpu_capacity_tokens=64 * 64, warmup_s=0.0)
    gpu_only = simulate(sessions, SimParams(**common, routing="prefix_load"), seed)
    tiered = simulate(sessions, SimParams(**common, routing="tiered_prefix_load"), seed)
    assert gpu_only["miss_routing_share"] > 0 and gpu_only["cpu_loaded_share"] == 0
    assert tiered["miss_routing_share"] == 0 and tiered["cpu_loaded_share"] > 0


def test_plan_adds_one_axis_per_trace_when_several_traces_are_given() -> None:
    from maxionbench.kvsim.__main__ import plan

    spec = {"traces": {"full": "a.jsonl", "mask": "b.jsonl"}, "matrix": {"capacity_tokens": [1, 2]}, "repeats": 2, "seed": 3}
    trials = plan(spec)
    assert len(trials) == 2 * 2 * 2
    assert trials[0][:3] == ("trace=full/capacity_tokens=1", {"trace": "full", "capacity_tokens": 1}, 0)
    assert {t[1]["trace"] for t in trials} == {"full", "mask"}
