from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from maxionbench.rag.answer_metrics import exact_match, normalize_answer, token_f1
from maxionbench.rag.fusion import reciprocal_rank_fusion
from maxionbench.rag.llm_client import CompletionResult, chat_completion
from maxionbench.rag.loadgen import RequestSpec, poisson_arrivals, run_open_loop, summarize
from maxionbench.rag.routing import LeastOutstanding, PrefixAffinity, RoundRobin


def test_answer_metrics_follow_squad_normalization() -> None:
    assert normalize_answer("The  Chief of Protocol!") == "chief of protocol"
    assert exact_match("the Animorphs", "Animorphs") == 1.0
    assert token_f1("Chief of Protocol of the US", "Chief of Protocol") == pytest.approx(0.75)
    assert token_f1("", "yes") == 0.0


def test_rrf_rewards_agreement_and_breaks_ties_stably() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], top_k=3)
    assert fused[:2] == ["b", "a"]
    assert reciprocal_rank_fusion([["x"], ["y"]], top_k=2) == ["x", "y"]


def test_round_robin_and_least_outstanding() -> None:
    rr = RoundRobin(3)
    assert [rr.acquire("k") for _ in range(4)] == [0, 1, 2, 0]
    lo = LeastOutstanding(2)
    assert lo.acquire("a") == 0
    assert lo.acquire("b") == 1
    lo.release(1)
    assert lo.acquire("c") == 1


def test_prefix_affinity_is_sticky_until_imbalanced() -> None:
    picker = PrefixAffinity(3, max_imbalance=1)
    home = picker.acquire("session-1")
    picker.release(home)
    assert picker.acquire("session-1") == home
    assert picker.acquire("session-1") == home
    spilled = picker.acquire("session-1")  # home now 2 ahead of the idle replicas
    assert spilled != home


def test_prefix_affinity_only_remaps_keys_from_failed_replica() -> None:
    picker = PrefixAffinity(3, max_imbalance=100)
    keys = [f"s{i}" for i in range(60)]
    before = {}
    for key in keys:
        before[key] = picker.acquire(key)
        picker.release(before[key])
    picker.mark_down(0)
    for key in keys:
        idx = picker.acquire(key)
        picker.release(idx)
        assert idx != 0
        if before[key] != 0:
            assert idx == before[key]


def test_poisson_arrivals_are_seeded_and_increasing() -> None:
    a = poisson_arrivals(50, 10.0, seed=7)
    assert a == poisson_arrivals(50, 10.0, seed=7)
    assert all(x < y for x, y in zip(a, a[1:]))
    assert 2.0 < a[-1] < 10.0


def _spec(i: int) -> RequestSpec:
    return RequestSpec(f"r{i}", f"s{i % 2}", f"s{i % 2}", ({"role": "user", "content": "q"},), "fallback")


def test_open_loop_admission_control_and_slo_accounting() -> None:
    def slow_send(url: str, messages: object, *, max_tokens: int, timeout_s: float) -> CompletionResult:
        time.sleep(0.2)
        return CompletionResult("ans", "ok", 0.05, 0.2, 100, 40, 3)

    records, duration = run_open_loop(
        [_spec(i) for i in range(6)],
        base_urls=["http://a", "http://b"],
        picker=RoundRobin(2),
        rate_rps=200.0,
        max_in_flight=2,
        timeout_s=1.0,
        max_tokens=8,
        seed=1,
        send=slow_send,
    )
    statuses = [r.status for r in records]
    assert statuses.count("ok") == 2
    assert statuses.count("rejected") == 4
    assert all(r.degraded for r in records if r.status == "rejected")
    summary = summarize(records, duration_s=duration, ttft_slo_s=1.0, e2e_slo_s=1.0)
    assert summary["status_counts"] == {"ok": 2, "rejected": 4}
    assert summary["prefix_cache_hit_ratio"] == pytest.approx(0.4)
    assert summary["slo_attainment"] == pytest.approx(2 / 6, abs=1e-4)


def test_open_loop_marks_endpoint_down_on_transport_error() -> None:
    def flaky_send(url: str, messages: object, *, max_tokens: int, timeout_s: float) -> CompletionResult:
        if url == "http://dead":
            return CompletionResult("", "error", None, 0.0, 0, 0, 0, "ConnectionRefusedError: refused")
        return CompletionResult("ans", "ok", 0.01, 0.01, 10, 0, 1)

    picker = RoundRobin(2)
    records, _ = run_open_loop(
        [_spec(i) for i in range(6)],
        base_urls=["http://dead", "http://live"],
        picker=picker,
        rate_rps=20.0,
        max_in_flight=4,
        timeout_s=1.0,
        max_tokens=8,
        seed=3,
        send=flaky_send,
    )
    assert picker.healthy == [False, True]
    assert [r.status for r in records].count("error") == 1
    assert [r.status for r in records].count("ok") == 5


class _FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        assert body["stream"] is True
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"content": "Par"}}]},
            {"choices": [{"delta": {"content": "is"}}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 30}}},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:
        pass


def test_chat_completion_parses_stream_usage_and_ttft() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = chat_completion(
            f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "hi"}], max_tokens=4, timeout_s=5
        )
    finally:
        server.shutdown()
    assert result.status == "ok"
    assert result.text == "Paris"
    assert (result.prompt_tokens, result.cached_tokens, result.completion_tokens) == (50, 30, 2)
    assert result.ttft_s is not None and result.ttft_s <= result.e2e_s


def test_chat_completion_reports_connection_failure_without_raising() -> None:
    result = chat_completion("http://127.0.0.1:9", [{"role": "user", "content": "hi"}], max_tokens=4, timeout_s=2)
    assert result.status == "error"
    assert not (result.error or "").startswith("http ")


def test_serving_workload_shares_session_prefix_and_preserves_turn_order(tmp_path) -> None:
    from maxionbench.tools.serving_bench import build_workload

    (tmp_path / "corpus.jsonl").write_text(
        "".join(json.dumps({"doc_id": f"d{i}", "text": f"paragraph {i}"}) + "\n" for i in range(20)), encoding="utf-8"
    )
    (tmp_path / "queries.jsonl").write_text(
        "".join(json.dumps({"query_id": f"q{i}", "text": f"question {i}", "answer": "a"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    (tmp_path / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(5)), encoding="utf-8"
    )
    specs = build_workload(tmp_path, sessions=4, turns=3, k=3, window=2, seed=0)
    assert len(specs) == 12
    assert specs == build_workload(tmp_path, sessions=4, turns=3, k=3, window=2, seed=0)
    by_session: dict[str, list[RequestSpec]] = {}
    for spec in specs:
        by_session.setdefault(spec.session_id, []).append(spec)
    for turns in by_session.values():
        assert [s.request_id.split("-t")[1] for s in turns] == ["0", "1", "2"]
        contexts = {s.messages[1]["content"].rsplit("\n\nQuestion:", 1)[0] for s in turns}
        assert len(contexts) == 1  # identical prefix across turns -> cacheable
        assert len({s.prefix_key for s in turns}) == 1


def test_paired_generation_deltas_pairs_by_query() -> None:
    from maxionbench.rag.stats import paired_generation_deltas

    recs = [
        {"status": "ok", "pipeline": p, "k": 3, "query_id": f"q{i}", "em": em, "f1": em}
        for i in range(20)
        for p, em in (("dense", 0.0), ("rerank", 1.0 if i < 15 else 0.0))
    ]
    out = paired_generation_deltas(recs, [("rerank@k3", "dense@k3"), ("missing@k3", "dense@k3")])
    assert list(out) == ["rerank@k3 - dense@k3"]
    em = out["rerank@k3 - dense@k3"]["em"]
    assert (em["n"], em["delta"], em["wins"], em["losses"]) == (20, 0.75, 15, 0)
    assert em["ci95"][0] > 0
