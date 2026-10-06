"""Quality-evaluation calls (not load tests): chat requests whose answer text is kept.

For paid targets the worst-case cost is reserved in the shared ledger before any request and the
provider-reported usage is committed afterwards, as the load-test runner does. `metered` exposes the
same accounting to callers that issue requests themselves (the agent loop).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from maxionbench.harness.budget import BudgetLedger, ModelPrice, cost_usd
from maxionbench.harness.targets import Target
from maxionbench.rag.llm_client import CompletionResult, chat_completion

LedgerFactory = Callable[[float], BudgetLedger]


@dataclass(frozen=True)
class Call:
    id: str
    messages: tuple[Mapping[str, Any], ...]
    max_tokens: int
    tools: tuple[Mapping[str, Any], ...] = ()  # OpenAI tools; needs a tool-capable `send`


def estimated_prompt_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Conservative (~3 chars/token plus per-message overhead), matching the runner's estimator."""
    return sum(len(str(m.get("content") or "")) // 3 + 8 for m in messages)


@dataclass
class Meter:
    """Usage of every request sent under one reservation; failures are charged their estimate."""

    price: ModelPrice | None
    usage: dict[str, int] = field(
        default_factory=lambda: {"requests": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                                 "estimated_requests": 0})
    override_usd: float | None = None  # set when the cost is not plain token pricing (storage, batch rates)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, result: CompletionResult, messages: Sequence[Mapping[str, Any]], max_tokens: int) -> None:
        with self._lock:
            self.usage["requests"] += 1
            if result.status == "ok" and result.prompt_tokens > 0:
                self.usage["input_tokens"] += result.prompt_tokens
                self.usage["cached_tokens"] += result.cached_tokens
                self.usage["output_tokens"] += result.completion_tokens
            elif (result.error or "").startswith("http "):
                pass  # provider rejected the request: not billed (as in the Go gateway)
            else:  # timeout or transport failure: may have been billed without usage data
                self.usage["input_tokens"] += estimated_prompt_tokens(messages)
                self.usage["output_tokens"] += max_tokens
                self.usage["estimated_requests"] += 1

    @property
    def spend_usd(self) -> float:
        if self.override_usd is not None:
            return self.override_usd
        if self.price is None:
            return 0.0
        u = self.usage
        return cost_usd(self.price, input_tokens=u["input_tokens"], output_tokens=u["output_tokens"],
                        cached_tokens=u["cached_tokens"])


@contextmanager
def metered(
    target: Target, estimate_usd: Callable[[ModelPrice], float], label: str,
    ledger_factory: LedgerFactory = BudgetLedger,
) -> Iterator[Meter]:
    """Reserve `estimate_usd(price)` for a paid target, then commit the meter's actual usage."""
    pricing = target.pricing()
    if pricing is None:
        yield Meter(None)
        return
    _, price, cap = pricing
    ledger = ledger_factory(cap)
    reservation = ledger.reserve(estimate_usd(price), label)
    meter = Meter(price)
    try:
        yield meter
    except BaseException:
        ledger.commit(reservation, max(meter.spend_usd, reservation.estimate_usd),
                      {**meter.usage, "note": "run failed; at least the estimate charged"})
        raise
    ledger.commit(reservation, meter.spend_usd, meter.usage)


def bound_send(target: Target, send: Callable[..., CompletionResult], extra_body: Mapping[str, Any] | None = None,
               ) -> Callable[..., CompletionResult]:
    """`send` with the target's request options (model, auth, path) applied."""
    options = target.request_options()
    options["extra_body"] = {**options.get("extra_body", {}), **(extra_body or {})}
    return functools.partial(send, **options)


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
    """Run `calls` against a started target; returns results by call id and a usage/spend summary."""
    fn = bound_send(target, send, extra_body)

    def estimate(price: ModelPrice) -> float:  # retried requests may bill again
        return (1 + retries) * sum(cost_usd(price, input_tokens=estimated_prompt_tokens(c.messages),
                                            output_tokens=c.max_tokens) for c in calls)

    with metered(target, estimate, label, ledger_factory) as meter:
        def one(call: Call) -> CompletionResult:
            kwargs = {"tools": list(call.tools)} if call.tools else {}
            for attempt in range(retries + 1):
                result = fn(target.base_urls[0], call.messages, max_tokens=call.max_tokens, timeout_s=timeout_s,
                            **kwargs)
                meter.add(result, call.messages, call.max_tokens)
                if result.status == "ok" or not retryable(result.error):
                    break
                time.sleep(2.0 * (attempt + 1))
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(zip((c.id for c in calls), pool.map(one, calls)))
    return results, {**meter.usage, "spend_usd": round(meter.spend_usd, 6)}


def retryable(error: str | None) -> bool:
    error = error or ""
    return error.startswith(("http 429", "http 500", "http 502", "http 503", "http 504")) or "timeout" in error
