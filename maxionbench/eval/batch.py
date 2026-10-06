"""Quality-evaluation calls (not load tests): many chat requests whose answer text is kept.

For paid targets the worst-case cost is reserved in the shared ledger before any request and the
provider-reported usage is committed afterwards, as the load-test runner does.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
import time
from typing import Any, Callable, Mapping, Sequence

from maxionbench.harness.budget import BudgetLedger, cost_usd
from maxionbench.harness.targets import Target
from maxionbench.rag.llm_client import CompletionResult, chat_completion

LedgerFactory = Callable[[float], BudgetLedger]


@dataclass(frozen=True)
class Call:
    id: str
    messages: tuple[Mapping[str, str], ...]
    max_tokens: int


def estimated_prompt_tokens(call: Call) -> int:
    """Conservative (~3 chars/token plus per-message overhead), matching the runner's estimator."""
    return sum(len(m.get("content", "")) // 3 + 8 for m in call.messages)


def run_calls(
    target: Target,
    calls: Sequence[Call],
    label: str,
    *,
    workers: int = 4,
    timeout_s: float = 60.0,
    retries: int = 2,
    extra_body: Mapping[str, Any] | None = None,
    send: Callable[..., CompletionResult] = chat_completion,
    ledger_factory: LedgerFactory = BudgetLedger,
) -> tuple[dict[str, CompletionResult], dict[str, Any]]:
    """Run `calls` against a started target; returns results by call id and a spend summary."""
    options = target.request_options()
    options["extra_body"] = {**options.get("extra_body", {}), **(extra_body or {})}
    fn = functools.partial(send, **options)
    pricing = target.pricing()
    ledger = reservation = None
    if pricing is not None:
        _, price, cap = pricing
        estimate = sum(cost_usd(price, input_tokens=estimated_prompt_tokens(c), output_tokens=c.max_tokens)
                       for c in calls)
        ledger = ledger_factory(cap)
        reservation = ledger.reserve(estimate * (1 + retries), label)  # retried requests may bill twice

    def one(call: Call) -> CompletionResult:
        for attempt in range(retries + 1):
            result = fn(target.base_urls[0], call.messages, max_tokens=call.max_tokens, timeout_s=timeout_s)
            if result.status == "ok" or not _retryable(result.error):
                return result
            time.sleep(2.0 * (attempt + 1))
        return result

    results: dict[str, CompletionResult] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for call, result in zip(calls, pool.map(one, calls)):
                results[call.id] = result
    except BaseException:
        if ledger is not None and reservation is not None:
            ledger.commit(reservation, reservation.estimate_usd, {"note": "batch failed; estimate charged"})
        raise
    usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "estimated_requests": 0}
    for call in calls:
        r = results[call.id]
        if r.status == "ok" and r.prompt_tokens > 0:
            usage["input_tokens"] += r.prompt_tokens
            usage["cached_tokens"] += r.cached_tokens
            usage["output_tokens"] += r.completion_tokens
        else:  # may have been billed without usage data: charge the estimate
            usage["input_tokens"] += estimated_prompt_tokens(call)
            usage["output_tokens"] += call.max_tokens
            usage["estimated_requests"] += 1
    spend = 0.0
    if ledger is not None and reservation is not None and pricing is not None:
        spend = cost_usd(pricing[1], input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                         cached_tokens=usage["cached_tokens"])
        ledger.commit(reservation, spend, usage)
    return results, {**usage, "spend_usd": round(spend, 6)}


def _retryable(error: str | None) -> bool:
    error = error or ""
    return error.startswith(("http 429", "http 500", "http 502", "http 503", "http 504")) or "timeout" in error
