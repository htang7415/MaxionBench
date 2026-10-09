from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from typing import Any

import pytest

from maxionbench.agents.browsecomp_env import BrowseTask
from maxionbench.agents.context import MASK_TEXT
from maxionbench.eval import context_eval
from maxionbench.eval.batch import Meter
from maxionbench.harness.budget import BudgetLedger, ModelPrice
from maxionbench.harness.gateway import AIGateway
from maxionbench.rag.llm_client import CompletionResult

PRICE = ModelPrice(input_per_m=1.0, output_per_m=10.0, cached_input_per_m=0.1, cache_storage_per_m_hour=0.0,
                   batch_input_per_m=0.5, batch_output_per_m=5.0)
TASK = BrowseTask("7", "Which mill?", "paper",
                  {f"d{i}": f"page {i} about the paper mill " + "word " * 3_000 for i in range(4)}, ("d0",))


class FakeModel:
    """Searches three times, then answers; records every prompt it is sent."""

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, Any]]] = []

    def tools(self, base_url: str, messages: Any, *, max_tokens: int, timeout_s: float, tools: Any) -> CompletionResult:
        self.prompts.append(list(messages))
        step = len(self.prompts)
        if step <= 3:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{step}", "type": "function", "function": {"name": "search", "arguments": '{"query": "mill"}'}}]}
        else:
            message = {"role": "assistant", "content": "paper"}
        return CompletionResult(json.dumps(message), "ok", None, 0.1, 1_000 * step, 500 * step, 10)

    def text(self, base_url: str, messages: Any, *, max_tokens: int, timeout_s: float) -> CompletionResult:
        return CompletionResult("summary of work", "ok", None, 0.1, 2_000, 0, 50)


def with_retries(fn: Any, meter: Meter, messages: Any, max_tokens: int, **kwargs: Any) -> Any:
    r = fn("http://unused", messages, max_tokens=max_tokens, timeout_s=1.0, **kwargs)
    meter.add(r, messages, max_tokens)
    return r


@pytest.mark.parametrize("name,params", [("full", {}), ("mask", {"keep": 1}),
                                         ("summarize", {"trigger_tokens": 3_000, "keep": 1})])
def test_run_one_records_cost_and_isolates_each_run(tmp_path: Path, name: str, params: dict[str, Any]) -> None:
    model = FakeModel()
    item, answer = context_eval.run_one(TASK, name, params, 6, tmp_path, Meter(PRICE), PRICE, with_retries,
                                        model.tools, model.text)
    assert item["run_status"] == "answered" and answer == "paper" and item["model_calls"] == 4
    assert "paper" not in json.dumps(item)  # items hold ids and numbers only
    assert model.prompts[0][0]["content"].startswith("run ")  # unique run id leads the system prompt
    agent_cost = sum((1_000 * s - 500 * s) * 1.0 + 500 * s * 0.1 + 10 * 10.0 for s in range(1, 5)) / 1e6
    assert item["cost_usd"] == pytest.approx(agent_cost + item["summary_cost_usd"], abs=1e-6)
    if name == "full":
        assert item["view_share"] == 1.0 and item["summary_calls"] == 0
    if name == "mask":
        assert item["view_share"] < 1.0
        assert sum(m.get("content") == MASK_TEXT for m in model.prompts[-1]) == 2  # 3 searches, newest kept
    if name == "summarize":
        assert item["summary_calls"] >= 1 and item["summary_cost_usd"] > 0
        assert any(str(m.get("content", "")).startswith("Summary of earlier work") for m in model.prompts[-1])


def test_metrics_and_paired_differences() -> None:
    def item(task: str, policy: str, correct: bool, cost: float) -> dict[str, Any]:
        return {"task_id": task, "policy": policy, "correct": correct, "cost_usd": cost, "prompt_tokens": 100,
                "cached_tokens": 50, "peak_context_tokens": 80, "model_calls": 4, "view_share": 1.0,
                "summary_cost_usd": 0.0, "run_status": "answered"}

    items = [item("a", "full", True, 0.04), item("b", "full", False, 0.02), item("c", "full", True, 0.03),
             item("a", "mask", True, 0.02), item("b", "mask", True, 0.01), item("c", "mask", False, 0.02)]
    m = context_eval.metrics([i for i in items if i["policy"] == "mask"])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["usd_per_correct"] == pytest.approx(0.025)
    paired = context_eval.paired_vs_full(items, ["full", "mask"])["mask"]
    assert paired["tasks"] == 3 and paired["correct"]["mean"] == 0.0
    assert paired["cost_usd"]["mean"] == pytest.approx(-0.04 / 3, abs=1e-6)


def test_transcript_renders_calls_and_results_as_text() -> None:
    text = context_eval.transcript([
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "content": "page"},
    ])
    assert text == "[user] task\n[agent called search({})]\n[tool result] page"


class FakeTarget:
    base_urls = ["http://unused"]

    def __init__(self, params: Any) -> None:
        pass

    def __enter__(self) -> "FakeTarget":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def pricing(self) -> tuple[str, ModelPrice, float]:
        return "fake", PRICE, 100.0

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake"}


def test_run_stops_on_refused_credits_and_resume_finishes_only_missing_tasks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [BrowseTask(str(i), f"Which mill {i}?", "paper", dict(TASK.docs), ("d0",)) for i in range(3)]
    state: dict[str, Any] = {"refuse_after": None, "calls": 0, "judged": []}

    def tools(base_url: str, messages: Any, *, max_tokens: int, timeout_s: float, tools: Any) -> CompletionResult:
        state["calls"] += 1
        if state["refuse_after"] is not None and state["calls"] > state["refuse_after"]:
            return CompletionResult("", "error", None, 0.1, 0, 0, 0, "http 402: prepayment credits are depleted")
        searched = sum(m["role"] == "tool" for m in messages)
        message = ({"role": "assistant", "content": "paper"} if searched else
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "mill"}'}}]})
        return CompletionResult(json.dumps(message), "ok", None, 0.1, 1_000, 0, 10)

    def judge(target: Any, calls: Any, label: str, **kwargs: Any) -> Any:
        state["judged"] += [c.id for c in calls]
        return ({c.id: CompletionResult('{"label": "correct"}', "ok", None, 0.1, 10, 0, 5) for c in calls},
                {"spend_usd": 0.0})

    class FakeMeter:
        def __enter__(self) -> Meter:
            return Meter(PRICE)

        def __exit__(self, *exc: object) -> None:
            pass

    monkeypatch.setattr(context_eval, "load_tasks", lambda n, seed, pool, exclude: tasks[:n])
    monkeypatch.setattr(context_eval, "GeminiTarget", FakeTarget)
    monkeypatch.setattr(context_eval, "bound_send", lambda target, send: tools if send.__name__ == "chat_tools" else None)
    monkeypatch.setattr(context_eval, "metered", lambda *a, **k: FakeMeter())
    monkeypatch.setattr(context_eval, "run_calls", judge)
    spec = {"schema_version": context_eval.SCHEMA, "name": "t", "seed": 0, "pool": 1, "tasks": 3, "max_steps": 4,
            "shards": 3, "budget_usd": 10.0, "model": {}, "policies": {"full": {}, "mask": {"keep": 1}}}
    # each run makes 2 model calls (search, answer): tasks 0 and 1 finish (8 calls); task 2's first run is refused
    state["refuse_after"] = 9
    with pytest.raises(context_eval.CreditsExhausted, match="--resume"):
        context_eval.run_context_eval(spec, tmp_path)
    run_dir = next(tmp_path.iterdir())
    saved = [json.loads(line) for line in (run_dir / "items.jsonl").read_text().splitlines()]
    assert sorted({i["task_id"] for i in saved}) == ["0", "1"] and len(saved) == 4  # task 2's partial group dropped
    assert not (run_dir / "results.json").exists() and state["judged"] == []

    state["refuse_after"], calls_before = None, state["calls"]
    out = context_eval.run_context_eval({}, tmp_path, resume=run_dir)
    assert state["calls"] - calls_before == 4  # only task 2 reran (2 policies x 2 calls)
    assert len(state["judged"]) == 6
    summary = json.loads((out / "summary.json").read_text())
    assert summary["tasks"] == 3 and summary["overall"]["full"]["accuracy"] == 1.0
    answers = (out / "answers.jsonl").read_text()
    assert answers.count("paper") == 6 and "paper" not in (out / "results.json").read_text()


class _FakeGemini(BaseHTTPRequestHandler):
    """Non-streaming provider: three searches, then an answer; records each body it receives."""
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).bodies.append(body)
        step = len(type(self).bodies)
        message = ({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{step}", "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "mill"}'}}]} if step <= 3 else
                   {"role": "assistant", "content": "paper"})
        raw = json.dumps({"choices": [{"message": message}], "usage": {"prompt_tokens": 1_000, "completion_tokens": 10,
                                                                         "total_tokens": 1_010}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain not installed")
def test_gateway_arm_trims_in_the_gateway_and_bills_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "FAKE-context-eval-key-abcdefghijkl")
    monkeypatch.setenv("MAXIONBENCH_BUDGET_DIR", str(tmp_path / "budget"))
    remote = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGemini)
    threading.Thread(target=remote.serve_forever, daemon=True).start()
    _FakeGemini.bodies = []

    def retry(fn: Any, meter: Meter, messages: Any, max_tokens: int, url: str | None = None, **kwargs: Any) -> Any:
        r = fn(url, messages, max_tokens=max_tokens, timeout_s=10.0, **kwargs)
        meter.add(r, messages, max_tokens)
        return r

    params = {"gateway": {"policy": "window+cache", "keep": 1, "budget_tokens": 5_000}}
    gw = AIGateway({"policy": "remote_only", "port": 18091, "context": params["gateway"],
                    "remote": {"enabled": True, "model": "gemini-3.5-flash-lite", "base_url": f"http://127.0.0.1:{remote.server_port}"}},
                   tmp_path / "gw")
    try:
        with gw:
            meter = Meter(PRICE)
            item, answer = context_eval.run_one(TASK, "gw-window+cache", params, 6, tmp_path, meter, PRICE, retry,
                                                None, None, gateway_url=gw.base_urls[0])
            stats = context_eval.scrape_context_metrics(gw.base_urls[0])
    finally:
        remote.shutdown()
    assert answer == "paper" and item["model_calls"] == 4 and meter.usage["requests"] == 4
    assert all("prompt_cache_key" not in b for b in _FakeGemini.bodies)  # Gemini rejects the field
    # each search adds ~3k tokens: the gateway trims once the history passes the 5k budget
    assert len(_FakeGemini.bodies[-1]["messages"]) < 2 + 2 * 3
    assert sum(v for k, v in stats.items() if k.startswith("context_requests_total")) == 4
    assert BudgetLedger(10.0).spent_usd() == pytest.approx(4 * (1_000 * 0.30 + 10 * 2.50) / 1e6)
