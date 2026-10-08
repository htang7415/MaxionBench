"""Context policies: what an agent sends the model each step, given its full history.

History is OpenAI-style: a system message, the task (user), then exchanges, each an assistant message
(with tool calls or text) followed by its tool results. Policies work on whole exchanges, so a tool call
is never separated from its result. A policy is stateful (`view` is called once per step with the
history so far); `reset()` starts a new task.

Prefix caching rewards append-only views: a view that only grows keeps the previous call's prompt as a
cached prefix. `full` and `truncate` are append-only, `summarize` breaks the prefix only when it compacts,
`window` and `mask` change earlier messages every step. `CacheAware` keeps a base policy's view
append-only until it passes a token budget, then applies the base policy once.

Defaults come from the Copilot traces (Step 1): tool output is ~half of production context, results over
2k tokens carry ~2/3 of it, and production compacts at ~130k tokens.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

Message = dict[str, Any]
TokenCounter = Callable[[Sequence[Message]], int]
Summarizer = Callable[[Sequence[Message]], str]

MASK_TEXT = "[tool result omitted to save context; call the tool again if you need it]"
SUMMARY_PREFIX = "Summary of earlier work on this task:\n"


def estimate_tokens(messages: Sequence[Message]) -> int:
    """~4 characters per token plus per-message overhead; tool-call arguments count too. A message with a
    `tokens` field (metadata-only traces, e.g. the Copilot replay) counts that instead of its text."""
    total = 0
    for m in messages:
        total += int(m["tokens"]) if "tokens" in m else len(str(m.get("content") or "")) // 4 + 4
        for call in m.get("tool_calls") or ():
            total += len(str(call.get("function", {}).get("arguments") or "")) // 4 + 4
    return total


def split(history: Sequence[Message]) -> tuple[list[Message], list[list[Message]]]:
    """(head, exchanges): head is the system message and task; each exchange starts with an assistant
    message and holds the tool results that follow it."""
    head_len = 2 if len(history) > 1 and history[0]["role"] == "system" else 1
    head, exchanges = list(history[:head_len]), []
    for m in history[head_len:]:
        if m["role"] == "assistant" or not exchanges:
            exchanges.append([m])
        else:
            exchanges[-1].append(m)
    return head, exchanges


def _replace_content(m: Message, text: str) -> Message:
    """`m` with new text; a `tokens` count no longer applies."""
    return {**{k: v for k, v in m.items() if k != "tokens"}, "content": text}


def _flat(head: list[Message], exchanges: Sequence[Sequence[Message]]) -> list[Message]:
    return head + [m for ex in exchanges for m in ex]


class Policy(Protocol):
    name: str

    def view(self, history: Sequence[Message]) -> list[Message]: ...

    def reset(self) -> None: ...


class Full:
    name = "full"

    def view(self, history: Sequence[Message]) -> list[Message]:
        return list(history)

    def reset(self) -> None:
        pass


class Truncate:
    """Cut every tool result to its first `max_tokens` (append-only: a result is cut once, when it arrives)."""

    def __init__(self, max_tokens: int = 2_000) -> None:
        self.name = f"truncate{max_tokens}"
        self.max_chars = 4 * max_tokens

    def view(self, history: Sequence[Message]) -> list[Message]:
        out = []
        for m in history:
            text = str(m.get("content") or "")
            if m["role"] == "tool" and "tokens" in m and m["tokens"] > self.max_chars // 4:
                m = {**m, "content": f"{text}[truncated]", "tokens": self.max_chars // 4 + 8}
            elif m["role"] == "tool" and "tokens" not in m and len(text) > self.max_chars:
                cut = (len(text) - self.max_chars) // 4
                m = {**m, "content": f"{text[:self.max_chars]}\n[truncated: ~{cut} more tokens]"}
            out.append(m)
        return out

    def reset(self) -> None:
        pass


class Window:
    """Keep only the last `keep` exchanges."""

    def __init__(self, keep: int = 8) -> None:
        self.name = f"window{keep}"
        self.keep = keep

    def view(self, history: Sequence[Message]) -> list[Message]:
        head, exchanges = split(history)
        return _flat(head, exchanges[-self.keep:] if self.keep else [])

    def reset(self) -> None:
        pass


class Mask:
    """Replace tool results older than the last `keep` exchanges with a placeholder; actions stay."""

    def __init__(self, keep: int = 4) -> None:
        self.name = f"mask{keep}"
        self.keep = keep

    def view(self, history: Sequence[Message]) -> list[Message]:
        head, exchanges = split(history)
        old = len(exchanges) - self.keep
        return _flat(head, [
            [_replace_content(m, MASK_TEXT) if i < old and m["role"] == "tool" else m for m in ex]
            for i, ex in enumerate(exchanges)
        ])

    def reset(self) -> None:
        pass


class Summarize:
    """Above `trigger_tokens`, replace all but the last `keep` exchanges (and any earlier summary) with one
    summary message; later steps append to that compacted history until it passes the trigger again."""

    def __init__(self, summarizer: Summarizer, trigger_tokens: int = 64_000, keep: int = 2,
                 count: TokenCounter = estimate_tokens) -> None:
        self.name = f"summarize{trigger_tokens // 1000}k"
        self.summarizer, self.trigger, self.keep, self.count = summarizer, trigger_tokens, keep, count
        self.reset()

    def reset(self) -> None:
        self.summary: Message | None = None
        self.covered = 0  # exchanges replaced by the summary
        self.compactions = 0

    def _compose(self, head: list[Message], exchanges: list[list[Message]]) -> list[Message]:
        return _flat(head + ([self.summary] if self.summary else []), exchanges[self.covered:])

    def view(self, history: Sequence[Message]) -> list[Message]:
        head, exchanges = split(history)
        view = self._compose(head, exchanges)
        if self.count(view) > self.trigger and len(exchanges) - self.covered > self.keep:
            upto = len(exchanges) - self.keep
            older = ([self.summary] if self.summary else []) + _flat([], exchanges[self.covered:upto])
            self.summary = {"role": "user", "content": SUMMARY_PREFIX + self.summarizer(head + older)}
            self.covered = upto
            self.compactions += 1
            view = self._compose(head, exchanges)
        return view


class CacheAware:
    """Append-only view of a stateless `base`: new messages are appended as `base` renders them, and the
    whole view is re-rendered by `base` only when it passes `budget_tokens`.

    With `min_growth` > 0, a re-rendered view must also grow by `min_growth` tokens before the next re-render:
    when the base policy cannot bring a long history under the budget, re-rendering every call would break
    the cached prefix every call."""

    def __init__(self, base: Policy, budget_tokens: int = 64_000, count: TokenCounter = estimate_tokens,
                 min_growth: int = 0) -> None:
        self.name = f"{base.name}+cache{budget_tokens // 1000}k"
        self.base, self.budget, self.count, self.min_growth = base, budget_tokens, count, min_growth
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.prev: list[Message] | None = None
        self.seen = 0  # history messages already reflected in `prev`
        self.edits = 0
        self.trigger = self.budget

    def view(self, history: Sequence[Message]) -> list[Message]:
        base_view = self.base.view(history)
        if self.prev is not None:
            new = len(history) - self.seen  # the newest messages render the same in both views
            appended = self.prev + (base_view[-new:] if new else [])
            if self.count(appended) <= self.trigger:
                self.prev, self.seen = appended, len(history)
                return appended
            self.edits += 1
        self.prev, self.seen = base_view, len(history)
        if self.min_growth:
            self.trigger = max(self.budget, self.count(base_view) + self.min_growth)
        return base_view


def make_policy(name: str, summarizer: Summarizer | None = None, **params: Any) -> Policy:
    """`full`, `truncate`, `window`, `mask`, `summarize`; `<name>+cache` wraps it in CacheAware."""
    base_name, _, wrapper = name.partition("+")
    budget = params.pop("budget_tokens", 64_000)
    min_growth = params.pop("min_growth", 0)
    if base_name == "summarize":
        if summarizer is None:
            raise ValueError("summarize needs a summarizer")
        if wrapper:
            raise ValueError("summarize is stateful and already breaks the prefix only when it compacts")
        policy: Policy = Summarize(summarizer, **params)
    else:
        policy = {"full": Full, "truncate": Truncate, "window": Window, "mask": Mask}[base_name](**params)
    if wrapper == "cache":
        return CacheAware(policy, budget, min_growth=min_growth)
    if wrapper:
        raise ValueError(f"unknown wrapper {wrapper!r}")
    return policy
