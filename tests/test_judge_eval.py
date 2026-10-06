from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxionbench.eval.batch import Call, run_calls
from maxionbench.eval.judge_calibration import CALIBRATION_SET
from maxionbench.graders.judge import LABELS, agreement, cohen_kappa, judge_messages, parse_label
from maxionbench.harness.budget import BudgetLedger, ModelPrice
from maxionbench.harness.targets import Target
from maxionbench.rag.llm_client import CompletionResult


@pytest.mark.parametrize(
    "text, label",
    [
        ('{"label": "correct", "reason": "same"}', "correct"),
        ('```json\n{"label": "Missing", "reason": "abstains"}\n```', "missing"),
        ('{"label": "partly"}', None),
        ("correct", None),
    ],
)
def test_parse_label(text: str, label: str | None) -> None:
    assert parse_label(text) == label


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa(["correct", "missing"] * 5, ["correct", "missing"] * 5) == 1.0
    # observed 0.5 with both raters 50/50 on two labels -> expected 0.5 -> kappa 0
    assert cohen_kappa(["correct", "correct", "missing", "missing"], ["correct", "missing", "correct", "missing"]) == 0
    report = agreement(["correct", "incorrect", "missing"], ["correct", "missing", "missing"])
    assert report["accuracy"] == pytest.approx(2 / 3, abs=1e-4) and report["confusion"]["incorrect"]["missing"] == 1


def test_judge_prompt_lists_every_gold() -> None:
    system, user = judge_messages("q?", ["a", "b"], "ans")
    assert "Reply with JSON only" in system["content"] and "- a\n- b" in user["content"]


def test_calibration_set_is_labelled_and_balanced_enough() -> None:
    rows = [json.loads(line) for line in CALIBRATION_SET.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100 and len({r["id"] for r in rows}) == 100
    assert {r["reference_label"] for r in rows} == set(LABELS)
    assert all(r["answer_status"] == "ok" and r["golds"] for r in rows)


class _Paid(Target):
    base_urls = ["http://fake"]

    def pricing(self) -> tuple[str, ModelPrice, float]:
        return "m", ModelPrice(1.0, 10.0, 0.1, 0.0, 0.5, 5.0), 1.0


def test_run_calls_reserves_then_bills_usage_and_retries(tmp_path: Path) -> None:
    attempts: dict[str, int] = {}

    def send(url, messages, *, max_tokens, timeout_s, **kw):
        key = messages[0]["content"]
        attempts[key] = attempts.get(key, 0) + 1
        if key == "flaky" and attempts[key] == 1:
            return CompletionResult("", "error", None, 0.1, 0, 0, 0, "http 429: slow down")
        if key == "broken":
            return CompletionResult("", "error", None, 0.1, 0, 0, 0, "http 400: bad")
        return CompletionResult("ok", "ok", 0.1, 0.2, 1000, 0, 10)

    calls = [Call(k, ({"role": "user", "content": k},), max_tokens=20) for k in ("a", "flaky", "broken")]
    ledger_path = tmp_path / "ledger.jsonl"
    results, spend = run_calls(_Paid(), calls, "test", send=send, ledger_factory=lambda cap: BudgetLedger(cap, ledger_path))
    assert results["flaky"].status == "ok" and attempts == {"a": 1, "flaky": 2, "broken": 1}  # 400 is not retried
    assert spend["requests"] == 4 and spend["estimated_requests"] == 0  # HTTP errors are not billed
    billed = 2 * (1000 * 1.0 + 10 * 10.0) / 1e6
    assert spend["spend_usd"] == pytest.approx(billed, abs=1e-6)
    assert BudgetLedger(1.0, ledger_path).spent_usd() == pytest.approx(billed, abs=1e-6)


def test_metered_charges_timeouts_and_failed_runs(tmp_path: Path) -> None:
    from maxionbench.eval.batch import metered

    path = tmp_path / "ledger.jsonl"
    factory = lambda cap: BudgetLedger(cap, path)  # noqa: E731
    with metered(_Paid(), lambda price: 0.01, "t", factory) as meter:
        meter.add(CompletionResult("", "timeout", None, 1.0, 0, 0, 0, "socket timeout"),
                  [{"role": "user", "content": "x" * 30}], max_tokens=100)
    assert BudgetLedger(1.0, path).spent_usd() == pytest.approx((10 + 8) * 1.0 / 1e6 + 100 * 10.0 / 1e6)
    with pytest.raises(RuntimeError), metered(_Paid(), lambda price: 0.01, "t", factory):
        raise RuntimeError("boom")
    assert BudgetLedger(1.0, path).spent_usd() == pytest.approx(0.01 + 0.001018, abs=1e-6)
