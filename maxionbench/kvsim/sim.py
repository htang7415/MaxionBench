"""Offline KV-cache simulator: replay agent sessions at trace timing over N replicas.

Each replica holds a fixed number of 64-token KV blocks. A request reuses the longest leading run of
its prompt blocks that is resident on its replica (vLLM prefix caching), allocates the rest plus its
output, holds them for the trace's service time, then releases them. Released blocks stay cached
until evicted. Timing comes from the trace and does not react to hits, so policies are compared on
the same arrivals; the output is cache behaviour (hit rate, recomputed prefill tokens, why misses
happened), not latency.

Eviction (unreferenced blocks only):
- lru: least recently released first, tail blocks of a prompt first (vLLM's free-queue order).
- ewma: a block released by a stream is protected for `ewma_k` x that stream's average gap between
  calls; blocks whose protection lapsed go first (LRU), then the latest expected return.
- hint: protected for `hint_s` only if the stream's next call comes within `hint_s` (an ideal
  "paused for a tool, back soon" signal from the client or gateway).
- oracle: protected until the stream's true next call; an upper bound for pause-aware retention.
- predicted: like oracle, but the gap is the true gap times a log-normal error exp(N(0, noise_sigma)):
  how accurate a pause-length predictor (e.g. per tool) must be to recover the oracle's gain.
A uniform TTL for every block would order evictions exactly like LRU, so retention only helps when it
differs between sessions; that is what the three pause-aware policies test.

CPU tier (cpu_capacity_tokens > 0): a block evicted from the GPU tier is demoted to a per-replica
LRU tier in host memory (vLLM KV offloading / LMCache style) instead of dropped. A request then
reuses its GPU-resident prefix, loads the continuing run found in the CPU tier back to the GPU
(exclusive: it leaves the CPU tier), and recomputes only the rest. `load_cost` prices a loaded token
relative to a recomputed one (PCIe transfer vs prefill; an assumption, reported with the results).

Routing: round_robin (per request), session (sticky, random assignment, like prompt_cache_key
hashing), prefix_load (llm-d style: 3 x cached-prefix fraction + 2 x free in-flight capacity),
tiered_prefix_load (the same, counting the prefix held in the replica's CPU tier as cached).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
import math
import random
from typing import Sequence

from maxionbench.kvsim.traces import BLOCK_TOKENS, Request, Session

ROUTINGS = ("round_robin", "session", "prefix_load", "tiered_prefix_load")
EVICTIONS = ("lru", "ewma", "hint", "oracle", "predicted")
EWMA_ALPHA = 0.3
EWMA_PRIOR_S = 2.0  # median gap between calls in the AgentX corpus


@dataclass(frozen=True)
class SimParams:
    replicas: int
    concurrency: int  # sessions active at once; a finished session is replaced by the next one
    capacity_tokens: int  # KV capacity per replica
    routing: str = "prefix_load"
    eviction: str = "lru"
    ewma_k: float = 4.0
    hint_s: float = 60.0
    noise_sigma: float = 0.0
    warmup_s: float = 1800.0  # requests before this are not measured
    cpu_capacity_tokens: int = 0  # host-memory KV tier per replica; 0 = none
    load_cost: float = 0.2  # cost of loading a token from the CPU tier, relative to recomputing it

    def __post_init__(self) -> None:
        if self.routing not in ROUTINGS:
            raise ValueError(f"routing {self.routing!r} not in {ROUTINGS}")
        if self.eviction not in EVICTIONS:
            raise ValueError(f"eviction {self.eviction!r} not in {EVICTIONS}")
        if self.cpu_capacity_tokens < 0 or self.load_cost < 0:
            raise ValueError("cpu_capacity_tokens and load_cost must be non-negative")
        if self.replicas < 1 or self.concurrency < 1 or self.capacity_tokens < BLOCK_TOKENS:
            raise ValueError("replicas, concurrency, and capacity_tokens must be positive")


class FreeBlocks:
    """Cached, unreferenced blocks in eviction order: lapsed protection (LRU) first, then the protected
    block with the latest expected return. With no protection this is exactly LRU."""

    def __init__(self) -> None:
        self.idle: OrderedDict[int, None] = OrderedDict()
        self.protected: dict[int, int] = {}  # key -> seq of its live heap entries
        self.by_end: list[tuple[float, int, int]] = []  # (protection end, seq, key)
        self.by_return: list[tuple[float, int, int]] = []  # (-expected return, seq, key)
        self.seq = 0

    def __len__(self) -> int:
        return len(self.idle) + len(self.protected)

    def release(self, key: int, now: float, window: float, expected: float) -> None:
        if window <= 0:
            self.idle[key] = None
            return
        self.seq += 1
        self.protected[key] = self.seq
        heapq.heappush(self.by_end, (now + window, self.seq, key))
        heapq.heappush(self.by_return, (-expected, self.seq, key))

    def take(self, key: int) -> None:
        """A cached block was hit: it is referenced again and no longer evictable."""
        if key in self.idle:
            del self.idle[key]
        else:
            self.protected.pop(key, None)

    def evict(self, now: float) -> int | None:
        while self.by_end and self.by_end[0][0] <= now:
            _, seq, key = heapq.heappop(self.by_end)
            if self.protected.get(key) == seq:
                del self.protected[key]
                self.idle[key] = None
        if self.idle:
            return self.idle.popitem(last=False)[0]
        while self.by_return:
            _, seq, key = heapq.heappop(self.by_return)
            if self.protected.get(key) == seq:
                del self.protected[key]
                return key
        return None

    def compact(self) -> None:
        """Drop stale heap entries once they dominate (they are skipped lazily otherwise)."""
        live = len(self.protected)
        for name in ("by_end", "by_return"):
            heap = getattr(self, name)
            if len(heap) > 4 * live + 4096:
                kept = [e for e in heap if self.protected.get(e[2]) == e[1]]
                heapq.heapify(kept)
                setattr(self, name, kept)


class Replica:
    def __init__(self, capacity_blocks: int, cpu_blocks: int = 0) -> None:
        self.capacity = capacity_blocks
        self.cpu_capacity = cpu_blocks
        self.cpu: OrderedDict[int, None] = OrderedDict()  # host-memory tier, LRU
        self.ref: dict[int, int] = {}  # resident prompt blocks -> reference count
        self.resident = 0  # prompt blocks + in-flight output blocks
        self.free = FreeBlocks()
        self.inflight = 0
        self.requests = 0
        self.releases = 0

    def cached_prefix(self, keys: Sequence[int]) -> int:
        ref = self.ref
        n = 0
        for k in keys:
            if k not in ref:
                break
            n += 1
        return n

    def demote(self, key: int) -> None:
        if self.cpu_capacity:
            self.cpu[key] = None
            self.cpu.move_to_end(key)
            if len(self.cpu) > self.cpu_capacity:
                self.cpu.popitem(last=False)

    def cpu_run(self, keys: Sequence[int], start: int) -> int:
        """End index of the run of `keys` from `start` held in the CPU tier."""
        cpu = self.cpu
        j = start
        while j < len(keys) and keys[j] in cpu:
            j += 1
        return j


@dataclass
class _Stream:
    ewma: float = EWMA_PRIOR_S
    last_end: float | None = None


class _Totals:
    def __init__(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.hit_tokens = 0
        self.loaded_tokens = 0
        self.miss_cold = 0.0
        self.miss_evicted = 0.0
        self.miss_routing = 0.0
        self.overflow = 0


def simulate(sessions: Sequence[Session], p: SimParams, seed: int) -> dict[str, float]:
    """Replay `sessions` (shuffled by `seed`) with `p.concurrency` active at a time; returns metrics over
    requests arriving between `p.warmup_s` and the start of the last session."""
    rng = random.Random(seed)
    noise = random.Random(seed + 1)  # separate stream: predictor error must not change the session order
    order = list(range(len(sessions)))
    rng.shuffle(order)
    if len(order) <= p.concurrency:
        raise ValueError(f"need more than {p.concurrency} sessions to reach steady state, have {len(order)}")
    replicas = [Replica(p.capacity_tokens // BLOCK_TOKENS, p.cpu_capacity_tokens // BLOCK_TOKENS)
                for _ in range(p.replicas)]
    events: list[tuple[float, int, int, tuple]] = []  # (time, kind, seq, payload); kind 0 = done first
    seq = 0
    streams: dict[tuple[int, int], _Stream] = {}
    sticky: dict[int, int] = {}
    rr = 0
    seen: set[int] = set()
    tot = _Totals()
    evictions = 0
    pending = iter(order)
    last_start = math.inf

    def push(t: float, kind: int, payload: tuple) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, kind, seq, payload))

    def start(s_idx: int, t0: float) -> None:
        sticky[s_idx] = rng.randrange(p.replicas)
        s = sessions[s_idx]
        for r in s.requests:
            push(t0 + r.t, 2, (s_idx, t0, r))
        push(t0 + s.span, 1, (s_idx,))

    def route(s_idx: int, keys: list[int]) -> Replica:
        nonlocal rr
        if p.routing == "round_robin":
            rr += 1
            return replicas[rr % p.replicas]
        if p.routing == "session":
            return replicas[sticky[s_idx]]
        most = max(r.inflight for r in replicas) + 1
        n = max(len(keys), 1)

        def prefix(r: Replica) -> int:
            gpu = r.cached_prefix(keys)
            return r.cpu_run(keys, gpu) if p.routing == "tiered_prefix_load" else gpu

        best = max(range(p.replicas), key=lambda i: (
            3 * prefix(replicas[i]) / n + 2 * (1 - replicas[i].inflight / most), -replicas[i].inflight, -i))
        return replicas[best]

    for s_idx in (next(pending) for _ in range(p.concurrency)):
        start(s_idx, rng.uniform(0.0, p.warmup_s))

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if now >= last_start:
            break
        if kind == 1:  # session finished: start the next one
            nxt = next(pending, None)
            if nxt is None:
                last_start = now
            else:
                start(nxt, now)
            continue
        if kind == 0:  # request done: release its blocks
            rep, keys, out_blocks, window, expected = payload
            rep.inflight -= 1
            rep.resident -= out_blocks
            ref, free = rep.ref, rep.free
            for k in reversed(keys):  # tail first, so tails are evicted first
                c = ref[k] - 1
                ref[k] = c
                if c == 0:
                    free.release(k, now, window, expected)
            rep.releases += 1
            if rep.releases % 256 == 0:
                free.compact()
            continue
        s_idx, t0, req = payload
        req: Request
        base = s_idx << 32
        keys = [base + h for h in req.blocks.tolist()]
        rep = route(s_idx, keys)
        hit = rep.cached_prefix(keys)
        ref, free = rep.ref, rep.free
        for k in keys[:hit]:
            c = ref[k]
            if c == 0:
                free.take(k)
            ref[k] = c + 1
        loaded_end = rep.cpu_run(keys, hit)
        for k in keys[hit:loaded_end]:
            del rep.cpu[k]  # moves back to the GPU tier below
        missed = keys[loaded_end:]
        cold = routed = 0
        for k in missed:
            if k not in seen:
                cold += 1
                seen.add(k)
            elif any(k in other.ref or k in other.cpu for other in replicas if other is not rep):
                routed += 1
        out_blocks = -(-req.out_tokens // BLOCK_TOKENS)
        need = len(keys) - hit + out_blocks
        overflow = False
        while rep.resident + need > rep.capacity:
            victim = free.evict(now)
            if victim is None:
                overflow = True
                break
            del ref[victim]
            rep.demote(victim)
            rep.resident -= 1
            evictions += 1
        for k in keys[hit:]:
            c = ref.get(k)
            if c is None:
                ref[k] = 1
                rep.resident += 1
            else:  # resident but past a broken prefix: recomputed, shares the slot
                if c == 0:
                    free.take(k)
                ref[k] = c + 1
        rep.resident += out_blocks
        rep.inflight += 1
        rep.requests += 1
        if now >= p.warmup_s:
            hit_tokens = min(hit * BLOCK_TOKENS, req.in_tokens)
            loaded_tokens = min(loaded_end * BLOCK_TOKENS, req.in_tokens) - hit_tokens
            miss_tokens = req.in_tokens - hit_tokens - loaded_tokens
            tot.requests += 1
            tot.prompt_tokens += req.in_tokens
            tot.hit_tokens += hit_tokens
            tot.loaded_tokens += loaded_tokens
            if missed:
                per_block = miss_tokens / len(missed)
                tot.miss_cold += cold * per_block
                tot.miss_routing += routed * per_block
                tot.miss_evicted += (len(missed) - cold - routed) * per_block
            tot.overflow += overflow
        st = streams.setdefault((s_idx, req.stream), _Stream())
        if st.last_end is not None:
            st.ewma = EWMA_ALPHA * max(now - st.last_end, 0.0) + (1 - EWMA_ALPHA) * st.ewma
        end = now + req.dur
        st.last_end = end
        window, expected = _protection(p, st, end, t0 + req.next_t, noise)
        push(end, 0, (rep, keys, out_blocks, window, expected))

    if tot.requests == 0:
        raise ValueError("no requests in the measurement window; lower warmup_s or raise the session count")
    per_replica = [r.requests for r in replicas]
    mean_req = sum(per_replica) / len(per_replica)
    prompt = tot.prompt_tokens
    return {
        "requests_measured": float(tot.requests),
        "window_s": float(last_start - p.warmup_s) if math.isfinite(last_start) else float("nan"),
        "prompt_tokens_per_request": prompt / tot.requests,
        "token_hit_rate": tot.hit_tokens / prompt,
        "recomputed_tokens_per_request": (prompt - tot.hit_tokens - tot.loaded_tokens) / tot.requests,
        "cpu_loaded_share": tot.loaded_tokens / prompt,
        "prefill_cost_tokens_per_request":
            (prompt - tot.hit_tokens - (1 - p.load_cost) * tot.loaded_tokens) / tot.requests,
        "miss_cold_share": tot.miss_cold / prompt,
        "miss_evicted_share": tot.miss_evicted / prompt,
        "miss_routing_share": tot.miss_routing / prompt,
        "hit_rate_upper_bound": 1 - tot.miss_cold / prompt,
        "overflow_rate": tot.overflow / tot.requests,
        "evictions_per_request": evictions / max(sum(per_replica), 1),
        "load_imbalance": max(per_replica) / mean_req if mean_req else float("nan"),
    }


def _protection(p: SimParams, st: _Stream, end: float, next_arrival: float,
                noise: random.Random) -> tuple[float, float]:
    """(protection window, expected return) for blocks this request releases at `end`."""
    if p.eviction == "lru":
        return 0.0, end
    if p.eviction == "ewma":
        return p.ewma_k * st.ewma, end + st.ewma
    gap = next_arrival - end
    if p.eviction == "hint":
        return (p.hint_s, end + p.hint_s) if gap <= p.hint_s else (0.0, end)
    if not (math.isfinite(gap) and gap > 0):
        return 0.0, end
    if p.eviction == "predicted":
        gap *= math.exp(noise.gauss(0.0, p.noise_sigma))
    return gap, end + gap  # oracle, or predicted with error
