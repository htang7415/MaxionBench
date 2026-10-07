from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maxionbench.agents.browsecomp_env import BrowseTask
from maxionbench.agents.context import MASK_TEXT
from maxionbench.eval import context_eval
from maxionbench.eval.batch import Meter
from maxionbench.harness.budget import ModelPrice
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
    item = context_eval.run_one(TASK, name, params, 6, tmp_path, Meter(PRICE), PRICE, with_retries,
                                model.tools, model.text)
    assert item["run_status"] == "answered" and item["_text"] == "paper" and item["model_calls"] == 4
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
