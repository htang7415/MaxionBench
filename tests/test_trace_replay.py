from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from maxionbench.datasets import sources
from maxionbench.datasets.loaders import v03
from maxionbench.harness.workloads import make_workload
from maxionbench.rag.llm_client import CompletionResult
from maxionbench.rag.loadgen import run_open_loop
from maxionbench.rag.routing import PICKERS


def _fake_window(n: int = 200, duration_s: float = 20.0, seed: int = 0) -> v03.TraceWindow:
    rng = np.random.default_rng(seed)
    return v03.TraceWindow(
        arrival_s=np.sort(rng.uniform(0, duration_s, n)),
        context_tokens=rng.integers(100, 4000, n),
        generated_tokens=rng.integers(0, 600, n),
        duration_s=duration_s,
    )


def _replay_rate(arrivals: list[float], specs: list) -> tuple[float, list[int]]:
    """Run the open-loop driver with an instant fake server; returns dispatch rate and max_tokens sent."""
    sent: list[float] = []
    sent_max_tokens: list[int] = []
    lock = threading.Lock()

    def send(url, messages, *, max_tokens, timeout_s):
        with lock:
            sent.append(time.perf_counter())
            sent_max_tokens.append(max_tokens)
        return CompletionResult("", "ok", 0.001, 0.001, 10, 0, max_tokens)

    records, _ = run_open_loop(
        specs, base_urls=["http://x"], picker=PICKERS["round_robin"](1), rate_rps=None, max_in_flight=64,
        timeout_s=5, max_tokens=999, seed=0, send=send, arrivals=arrivals,
    )
    assert all(r.status == "ok" for r in records)
    sent.sort()
    return (len(sent) - 1) / (sent[-1] - sent[0]), sent_max_tokens


def test_trace_replay_scales_time_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _fake_window()
    monkeypatch.setattr(v03, "load_azure_trace", lambda start_s, duration_s: window)
    w = make_workload("trace_replay", {"duration_s": 20.0, "rate_scale": 0.5, "token_scale": 0.1,
                                       "max_output_tokens": 40}, seed=1)
    assert w.arrivals == pytest.approx((window.arrival_s / 0.5).tolist())
    assert [s.max_tokens for s in w.specs] == [min(40, max(1, round(g * 0.1))) for g in window.generated_tokens]
    words = [len(s.messages[0]["content"].split(": ", 1)[1].split()) for s in w.specs]
    assert words == [max(1, round(c * 0.1)) for c in window.context_tokens]
    assert len({s.prefix_key for s in w.specs}) == len(w.specs)  # no shared prefixes
    assert w.extra_body == {"ignore_eos": True} and w.rate_rps is None and w.concurrency is None
    assert make_workload("trace_replay", {"duration_s": 20.0, "rate_scale": 0.5}, seed=1).specs[0] != w.specs[0]
    with pytest.raises(ValueError, match="unknown params"):
        make_workload("trace_replay", {"duration_s": 1.0, "rate_rps": 3}, seed=1)


def test_replayed_rate_matches_trace_within_5pct(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _fake_window(n=300, duration_s=3.0)
    monkeypatch.setattr(v03, "load_azure_trace", lambda start_s, duration_s: window)
    w = make_workload("trace_replay", {"duration_s": 3.0, "rate_scale": 1.0, "token_scale": 0.01}, seed=0)
    rate, sent_max_tokens = _replay_rate(w.arrivals, w.specs)
    a = window.arrival_s
    trace_rate = (len(a) - 1) / (a[-1] - a[0])
    assert rate == pytest.approx(trace_rate, rel=0.05)
    assert sorted(sent_max_tokens) == sorted(s.max_tokens for s in w.specs)  # per-request lengths reach the server


@pytest.mark.skipif(not (sources.DATASET_ROOT / v03.AZURE_CONV_TRACE).exists(), reason="Azure trace not fetched")
def test_real_azure_window_replays_within_5pct() -> None:
    window = v03.load_azure_trace(0.0, 60.0)
    # 20x speed-up: 60 s of trace in ~3 s of wall time, so the test stays fast
    w = make_workload("trace_replay", {"start_s": 0.0, "duration_s": 60.0, "rate_scale": 20.0,
                                       "token_scale": 0.01}, seed=0)
    rate, _ = _replay_rate(w.arrivals, w.specs)
    a = window.arrival_s
    assert rate == pytest.approx(20.0 * (len(a) - 1) / (a[-1] - a[0]), rel=0.05)
