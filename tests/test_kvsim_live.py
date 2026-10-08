import math

import numpy as np
import pytest

from maxionbench.kvsim.live import Outcome, prompt_tokens, schedule, target_params, trial_metrics
from maxionbench.kvsim.traces import Request, Session


def _session(sid, times, span):
    reqs = tuple(Request(t=t, dur=1.0, blocks=np.arange(4, dtype=np.int32), in_tokens=256, out_tokens=64, stream=0,
                         next_t=math.inf) for t in times)
    return Session(id=sid, requests=reqs, span=span)


def test_prompt_tokens_share_prefixes_only_within_a_session():
    a = prompt_tokens(1, np.array([0, 1, 2], dtype=np.int32), 16)
    b = prompt_tokens(1, np.array([0, 1, 7], dtype=np.int32), 16)
    other = prompt_tokens(2, np.array([0, 1, 2], dtype=np.int32), 16)
    assert len(a) == 48 and a[:32] == b[:32] and a[32:] != b[32:]
    assert a[:16] != other[:16]
    assert all(1000 <= t < 150_000 for t in a + other)


def test_schedule_replaces_finished_sessions_and_scales_time():
    sessions = [_session(f"s{i}", [0.0, 40.0], span=41.0) for i in range(4)]
    arr = schedule(sessions, concurrency=2, horizon_s=30.0, stagger_s=0.0, time_scale=2.0, seed=0)
    # two sessions at t=0 (calls at 0 and 20), replaced at 20.5 by two more (calls at 20.5)
    assert [round(a.t, 2) for a in arr] == [0.0, 0.0, 20.0, 20.0, 20.5, 20.5]
    assert len({a.session for a in arr}) == 4
    assert schedule(sessions, 2, 30.0, 0.0, 2.0, seed=0) == arr  # deterministic


def test_trial_metrics_window_slo_and_cache_share():
    outcomes = [
        Outcome(scheduled_s=0.0, status="ok", ttft_s=9.0, e2e_s=9.0, prompt_tokens=100, cached_tokens=0),  # warm-up
        Outcome(scheduled_s=10.0, status="ok", ttft_s=0.5, e2e_s=1.0, prompt_tokens=100, cached_tokens=90),
        Outcome(scheduled_s=20.0, status="ok", ttft_s=2.0, e2e_s=3.0, prompt_tokens=100, cached_tokens=50),
        Outcome(scheduled_s=30.0, status="error", error="boom"),
    ]
    m = trial_metrics(outcomes, warmup_s=5.0, ttft_slo_s=1.0)
    assert m["requests_measured"] == 3 and m["error_rate"] == pytest.approx(1 / 3)
    assert m["slo_attainment"] == pytest.approx(1 / 3)
    assert m["cached_token_share"] == pytest.approx(0.7)
    assert m["recomputed_tokens_per_request"] == pytest.approx(30.0)
    assert m["goodput_rps"] == pytest.approx(1 / 25)


def test_target_params_set_kv_tiers_per_cell():
    target = {"replicas": 2, "image": "img", "gpu_kv_blocks": 100, "sim_args": ["--block-size", "16"]}
    p = target_params(target, {"scorer_profile": "precise-tiered", "cpu_kv_blocks": 400})
    assert p["scorer_profile"] == "precise-tiered" and p["worker_params"]["image"] == "img"
    assert p["worker_params"]["args"] == ["--block-size", "16", "--enable-kvcache", "--kv-cache-size", "100",
                                          "--cpu-kv-cache-size", "400"]


def test_precise_epp_config_uses_kv_events_and_tier_weights():
    from maxionbench.harness.llmd import PRECISE_PROFILES, render_epp_config

    for profile, cpu_weight in PRECISE_PROFILES.items():
        cfg = render_epp_config(profile, "qwen3", "http://192.168.65.254:8300", block_size=16)
        by_type = {p["type"]: p for p in cfg["plugins"]}
        producer = by_type["precise-prefix-cache-producer"]["parameters"]
        assert producer["kvEventsConfig"]["zmqEndpoint"] == "tcp://*:5557"
        assert producer["tokenProcessorConfig"]["blockSizeTokens"] == 16
        weights = {b["name"]: b["weight"] for b in producer["indexerConfig"]["kvCacheBackendConfigs"]}
        assert weights == {"gpu": 1.0, "cpu": cpu_weight}
        assert by_type["token-producer"]["parameters"]["vllm"]["url"] == "http://192.168.65.254:8300"
        assert by_type["prefix-cache-scorer"]["parameters"] == {
            "prefixMatchInfoProducerName": "precise-prefix-cache-producer"}
    with pytest.raises(ValueError, match="render_url"):
        render_epp_config("precise-tiered")


def test_target_params_supports_native_vllm_metal_workers() -> None:
    from maxionbench.kvsim.live import target_params

    target = {"workers": "vllm_metal", "model": "qwen3-0.6b",
              "worker_params": {"model": "~/models/Qwen3-0.6B-Q8_0.gguf", "replicas": 2, "max_model_len": 16384}}
    params = target_params(target, {"scorer_profile": "optimized-baseline"})
    assert params == {"workers": "vllm_metal", "scorer_profile": "optimized-baseline", "model": "qwen3-0.6b",
                      "worker_params": target["worker_params"]}
