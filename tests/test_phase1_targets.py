from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import pytest

from maxionbench.harness.budget import BudgetLedger, ModelPrice
from maxionbench.harness.planner import plan
from maxionbench.harness.runner import run_experiment
from maxionbench.harness.spec import parse_spec
from maxionbench.harness.targets import GeminiTarget, LlamaCppReplicas, StaticEndpoints, VllmMetal
from maxionbench.harness.workloads import make_workload
from maxionbench.rag.llm_client import chat_completion
from maxionbench.rag.loadgen import RequestSpec, run_closed_loop, summarize
from maxionbench.rag.routing import RoundRobin

FAKE_KEY = "FAKE-gemini-key-abcdefghijklmnop"


class _Recorder(BaseHTTPRequestHandler):
    """Streams 4 tokens with usage; records request bodies, headers, and peak concurrency."""

    bodies: list[dict[str, Any]] = []
    headers_seen: list[dict[str, str]] = []
    paths: list[str] = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
            cls.bodies.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            cls.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
            cls.paths.append(self.path)
        try:
            time.sleep(0.05)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for tok in ("a", "b", "c", "d"):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            usage = {"prompt_tokens": 1000, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 200}}
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with cls.lock:
                cls.active -= 1

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def recorder() -> Iterator[str]:
    _Recorder.bodies, _Recorder.headers_seen, _Recorder.paths = [], [], []
    _Recorder.active = _Recorder.peak = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def _specs(n: int) -> list[RequestSpec]:
    return [RequestSpec(f"r{i}", "s", "s", ({"role": "user", "content": "x" * 300},)) for i in range(n)]


def test_closed_loop_holds_concurrency_and_reports_tpot(recorder: str) -> None:
    records, duration = run_closed_loop(
        _specs(12), base_urls=[recorder], picker=RoundRobin(1), concurrency=3, timeout_s=5, max_tokens=8
    )
    assert len(records) == 12 and all(r.status == "ok" for r in records)
    assert _Recorder.peak == 3
    summary = summarize(records, duration_s=duration, ttft_slo_s=5, e2e_slo_s=5)
    assert summary["tpot"]["p50_ms"] > 0
    assert summary["output_tokens_per_s"] == pytest.approx(12 * 4 / duration, rel=0.01)


def test_spec_variants_drive_target_kind_and_params() -> None:
    spec = parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "engines",
            "repeats": 1,
            "target": {"kind": "static_endpoints", "params": {"urls": ["http://x"]}},
            "target_variants": {
                "cpu": {"kind": "llamacpp_replicas", "params": {"model": "m.gguf", "device": "cpu"}},
                "gpu": {"kind": "vllm_metal", "params": {"model": "m.gguf"}},
            },
            "workload": {"kind": "synthetic_chat", "params": {"concurrency": 1}},
            "slo": {"ttft_s": 1, "e2e_s": 2},
            "matrix": {"target.variant": ["cpu", "gpu"], "workload.concurrency": [1, 4]},
        }
    )
    trials = plan(spec)
    assert len(trials) == 4
    kinds = {t.cell_id: t.target_kind for t in trials}
    assert kinds["variant=cpu,concurrency=4"] == "llamacpp_replicas"
    assert kinds["variant=gpu,concurrency=1"] == "vllm_metal"
    gpu = next(t for t in trials if t.cell_id == "variant=gpu,concurrency=4")
    assert gpu.target_params == {"model": "m.gguf"} and gpu.workload_params["concurrency"] == 4
    assert spec.to_dict()["target_variants"]["cpu"]["kind"] == "llamacpp_replicas"
    with pytest.raises(ValueError, match="target_variants"):
        parse_spec({**spec.to_dict(), "matrix": {"target.variant": ["tpu"]}})


def test_workload_requires_exactly_one_load_mode() -> None:
    assert make_workload("synthetic_chat", {"concurrency": 2}, 0).concurrency == 2
    with pytest.raises(ValueError, match="exactly one"):
        make_workload("synthetic_chat", {"concurrency": 2, "rate_rps": 1.0}, 0)
    with pytest.raises(ValueError, match="exactly one"):
        make_workload("synthetic_chat", {}, 0)


def test_engine_commands_and_request_options(tmp_path: Path) -> None:
    model = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    model.write_bytes(b"gguf")
    cpu = LlamaCppReplicas({"model": str(model), "device": "cpu", "replicas": 1, "disable_thinking": True,
                            "cache_prompt": False}, tmp_path)
    metal = LlamaCppReplicas({"model": str(model), "device": "metal", "replicas": 1}, tmp_path)
    assert "--device" in cpu.command(0) and "none" in cpu.command(0)
    assert "-ngl" in metal.command(0) and "--device" not in metal.command(0)
    assert cpu.request_options()["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False,
    }
    with pytest.raises(ValueError, match="device"):
        LlamaCppReplicas({"model": str(model), "device": "tpu"}, tmp_path)
    vllm = VllmMetal({"model": str(model), "tokenizer": "Qwen/Qwen3-4B", "enable_prefix_caching": False}, tmp_path)
    cmd = vllm.command(0)
    assert cmd[1:3] == ["serve", str(model)]
    assert "--no-enable-prefix-caching" in cmd and cmd[cmd.index("--tokenizer") + 1] == "Qwen/Qwen3-4B"
    assert vllm.request_options()["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_gemini_target_sends_bearer_key_without_exposing_it(
    recorder: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    target = GeminiTarget({"model": "gemini-3.5-flash-lite", "reasoning_effort": "none"})
    options = target.request_options()
    assert FAKE_KEY not in repr(options) and FAKE_KEY not in json.dumps(target.describe())
    result = chat_completion(recorder, [{"role": "user", "content": "hi"}], max_tokens=4, timeout_s=5, **options)
    assert result.status == "ok"
    assert _Recorder.headers_seen[-1]["authorization"] == f"Bearer {FAKE_KEY}"
    assert _Recorder.paths[-1] == "/chat/completions"
    assert _Recorder.bodies[-1]["model"] == "gemini-3.5-flash-lite"
    assert _Recorder.bodies[-1]["reasoning_effort"] == "none"
    with pytest.raises(KeyError, match="no price"):
        GeminiTarget({"model": "gemini-unpriced"})


class _PaidFake(StaticEndpoints):
    PRICE = ModelPrice(input_per_m=1.0, output_per_m=10.0, cached_input_per_m=0.1,
                       cache_storage_per_m_hour=0.0, batch_input_per_m=0.5, batch_output_per_m=5.0)

    def __init__(self, params: dict[str, Any], cap: float) -> None:
        super().__init__(params)
        self.cap = cap

    def pricing(self) -> tuple[str, ModelPrice, float]:
        return "fake-model", self.PRICE, self.cap


def _paid_spec(url: str, requests: int) -> Any:
    return parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "paid",
            "repeats": 1,
            "target": {"kind": "static_endpoints", "params": {"urls": [url]}},
            "workload": {"kind": "synthetic_chat", "params": {"requests": requests, "prompt_words": 50,
                                                               "concurrency": 2, "max_tokens": 4}},
            "slo": {"ttft_s": 5, "e2e_s": 5},
            "quiet_host": {"max_load_1m": 1000, "wait_s": 0},
        }
    )


def test_paid_trial_commits_actual_cost_from_usage(recorder: str, tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _, result = run_experiment(
        _paid_spec(recorder, 5),
        tmp_path / "runs",
        target_factory=lambda kind, params, log_dir: _PaidFake(params, cap=10.0),
        ledger_factory=lambda cap: BudgetLedger(cap, ledger_path),
        log=lambda msg: None,
    )
    trial = result.trials[0]
    assert trial.status == "ok"
    # 5 requests x (800 uncached @ $1/M + 200 cached @ $0.1/M + 4 out @ $10/M)
    expected = 5 * (800 * 1.0 + 200 * 0.1 + 4 * 10.0) / 1e6
    assert trial.target["spend_usd"] == pytest.approx(expected)
    assert BudgetLedger(10.0, ledger_path).spent_usd() == pytest.approx(expected)
    assert BudgetLedger(10.0, ledger_path).remaining_usd() == pytest.approx(10.0 - expected)


def test_paid_trial_over_cap_sends_nothing(recorder: str, tmp_path: Path) -> None:
    _, result = run_experiment(
        _paid_spec(recorder, 50),
        tmp_path / "runs",
        target_factory=lambda kind, params, log_dir: _PaidFake(params, cap=0.0001),
        ledger_factory=lambda cap: BudgetLedger(cap, tmp_path / "ledger.jsonl"),
        log=lambda msg: None,
    )
    assert result.trials[0].status == "failed"
    assert "BudgetExceededError" in (result.trials[0].error or "")
    assert _Recorder.bodies == []  # refused before the first request


def test_warmup_requests_run_before_measurement_and_are_excluded(recorder: str, tmp_path: Path) -> None:
    spec = parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "warm",
            "repeats": 1,
            "target": {"kind": "static_endpoints", "params": {"urls": [recorder]}},
            "workload": {"kind": "synthetic_chat", "params": {"requests": 3, "concurrency": 1, "warmup_requests": 2}},
            "slo": {"ttft_s": 5, "e2e_s": 5},
            "quiet_host": {"max_load_1m": 1000, "wait_s": 0},
        }
    )
    _, result = run_experiment(spec, tmp_path, log=lambda msg: None)
    assert len(_Recorder.bodies) == 5
    assert [b["messages"][0]["content"].startswith("Warm-up") for b in _Recorder.bodies] == [True, True, False, False, False]
    assert result.trials[0].metrics["requests"] == 3.0


def test_paid_target_rejects_warmup(recorder: str, tmp_path: Path) -> None:
    payload = _paid_spec(recorder, 2).to_dict()
    payload["workload"]["params"]["warmup_requests"] = 1
    _, result = run_experiment(
        parse_spec(payload), tmp_path / "runs",
        target_factory=lambda kind, params, log_dir: _PaidFake(params, cap=10.0),
        ledger_factory=lambda cap: BudgetLedger(cap, tmp_path / "ledger.jsonl"),
        log=lambda msg: None,
    )
    assert "warmup_requests is not allowed" in (result.trials[0].error or "")
    assert _Recorder.bodies == []


class _CountingTarget(StaticEndpoints):
    starts = 0
    stops = 0

    def __enter__(self) -> "_CountingTarget":
        type(self).starts += 1
        return self

    def __exit__(self, *exc: object) -> None:
        type(self).stops += 1


def _reuse_spec(url: str, reuse: bool) -> Any:
    return parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "reuse",
            "repeats": 2,
            "reuse_targets": reuse,
            "target": {"kind": "static_endpoints", "params": {"urls": [url]}},
            "target_variants": {
                "a": {"kind": "static_endpoints", "params": {"urls": [url]}},
                "b": {"kind": "static_endpoints", "params": {"urls": [url], "routing_policy": "least_outstanding"}},
            },
            "workload": {"kind": "synthetic_chat", "params": {"requests": 2, "concurrency": 1, "ignore_eos": True,
                                                               "max_tokens": 4}},
            "slo": {"ttft_s": 5, "e2e_s": 5},
            "matrix": {"target.variant": ["a", "b"], "workload.concurrency": [1, 2, 3]},
            "quiet_host": {"max_load_1m": 1000, "wait_s": 0},
        }
    )


@pytest.mark.parametrize("reuse,expected_starts", [(True, 4), (False, 12)])
def test_reuse_targets_starts_once_per_group(recorder: str, tmp_path: Path, reuse: bool, expected_starts: int) -> None:
    _CountingTarget.starts = _CountingTarget.stops = 0
    spec = _reuse_spec(recorder, reuse)
    trials = plan(spec)
    if reuse:  # trials sharing a target are contiguous within each repeat
        for r in range(2):
            variants = [t.cell_params["target.variant"] for t in trials[r * 6 : r * 6 + 6]]
            assert variants[:3] == [variants[0]] * 3 and variants[3:] == [variants[3]] * 3
    _, result = run_experiment(
        spec, tmp_path, target_factory=lambda kind, params, log_dir: _CountingTarget(params), log=lambda m: None
    )
    assert all(t.status == "ok" for t in result.trials)
    assert _CountingTarget.starts == expected_starts == _CountingTarget.stops
    assert sum(t.target["reused"] for t in result.trials) == 12 - expected_starts
    assert all(b.get("ignore_eos") is True for b in _Recorder.bodies)


def test_failed_trial_closes_reused_target(recorder: str, tmp_path: Path) -> None:
    _CountingTarget.starts = _CountingTarget.stops = 0
    spec = _reuse_spec(recorder, True)
    calls = {"n": 0}

    class _FailSecond(_CountingTarget):
        def picker(self) -> Any:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("engine crashed mid-trial")
            return super().picker()

    _, result = run_experiment(
        spec, tmp_path, target_factory=lambda kind, params, log_dir: _FailSecond(params), log=lambda m: None
    )
    assert [t.status for t in result.trials].count("failed") == 1
    assert _FailSecond.starts == _FailSecond.stops == 5  # the crash forces one extra restart


def test_managed_target_refuses_busy_port(tmp_path: Path) -> None:
    import socket

    from maxionbench.harness.targets import port_in_use

    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        port = squatter.getsockname()[1]
        assert port_in_use(port)
        target = LlamaCppReplicas({"model": str(model), "replicas": 1, "base_port": port}, tmp_path)
        with pytest.raises(RuntimeError, match="already in use"):
            target.__enter__()
        assert target.procs == []  # nothing was launched


def test_llmd_epp_config_profiles_and_endpoints() -> None:
    from maxionbench.harness.llmd import SCORER_PROFILES, render_endpoints, render_epp_config

    for profile, scorers in SCORER_PROFILES.items():
        cfg = render_epp_config(profile)
        types = [p["type"] for p in cfg["plugins"]]
        assert types[0] == "file-discovery" and cfg["plugins"][0]["parameters"]["watchFile"] is True
        assert [s for s, _ in scorers] == [t for t in types if t.endswith("-scorer")]
        assert cfg["schedulingProfiles"][0]["plugins"] == [{"pluginRef": s, "weight": w} for s, w in scorers]
        assert cfg["dataLayer"]["discovery"] == {"pluginRef": "file-discovery"}
    with pytest.raises(ValueError, match="scorer_profile"):
        render_epp_config("round-robin-ish")
    eps = render_endpoints([8301, 8302], "qwen3")["endpoints"]
    assert [(e["address"], e["port"]) for e in eps] == [("192.168.65.254", "8301"), ("192.168.65.254", "8302")]
    assert all(e["labels"] == {"model": "qwen3"} for e in eps)


def test_llmd_target_wires_vllm_workers(tmp_path: Path) -> None:
    from maxionbench.harness.targets import make_target

    model = tmp_path / "Qwen3-1.7B-Q8_0.gguf"
    model.write_bytes(b"gguf")
    target = make_target(
        "llmd",
        {"workers": "vllm_metal", "scorer_profile": "prefix-aware", "model": "qwen3-1.7b",
         "worker_params": {"model": str(model), "replicas": 2, "base_port": 8210}},
        tmp_path,
    )
    assert target.base_urls == ["http://127.0.0.1:8081"]
    assert target.worker_ports == [8210, 8211]
    cmd = target.workers.command(1)
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen3-1.7b" and "8211" in cmd
    body = target.request_options()["extra_body"]
    assert body["model"] == "qwen3-1.7b" and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert target.describe()["scorer_profile"] == "prefix-aware"
    with pytest.raises(ValueError, match="workers"):
        make_target("llmd", {"workers": "tpu"}, tmp_path)
