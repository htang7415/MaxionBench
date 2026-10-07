"""Agent loop over an MCP tool server, with OpenAI-style messages so any chat model can be the policy.

A policy maps (messages, tools) to tool calls or a final answer. `OraclePolicy` is a scripted
agent that knows the gold paragraphs but must still find each one through `search` and `read`;
it checks that every task is solvable with the tools.

    python -m maxionbench.agents.loop --tasks 50   # oracle validation on real HotpotQA
"""

from __future__ import annotations

from argparse import ArgumentParser
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from maxionbench.agents.context import Policy as ContextPolicy
from maxionbench.agents.hotpot_env import DEFAULT_DATASET, AgentTask, HotpotCorpus, build_tasks
from maxionbench.rag.answer_metrics import normalize_answer

SYSTEM_PROMPT = (
    "Answer the question using the search and read tools. Questions need facts from more than one "
    "paragraph: search, read, then search again for what you learned. Reply with only the short final "
    "answer (a name, number, date, or yes/no) when done."
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PolicyOutput:
    tool_calls: tuple[ToolCall, ...] = ()
    answer: str | None = None
    assistant_message: dict[str, Any] | None = None  # model's own message, replayed verbatim if given


Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]]], PolicyOutput]


@dataclass
class AgentRun:
    task_id: str
    status: str  # "answered" | "max_steps" | "error"
    answer: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"name", "arguments"} in order
    error: str | None = None


def openai_tools(mcp_tools: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
        for t in mcp_tools
    ]


async def run_agent(session: ClientSession, policy: Policy, task: AgentTask, max_steps: int = 8,
                    system_prompt: str = SYSTEM_PROMPT, context: ContextPolicy | None = None) -> AgentRun:
    """Run `task`; the model sees `context.view(history)` each step (the full history by default)."""
    tools = openai_tools((await session.list_tools()).tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.question},
    ]
    run = AgentRun(task.task_id, "max_steps")
    try:
        for _ in range(max_steps):
            out = policy(context.view(messages) if context else messages, tools)
            if out.answer is not None:
                run.status, run.answer = "answered", out.answer
                return run
            messages.append(out.assistant_message or {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in out.tool_calls
                ],
            })
            for call in out.tool_calls:
                run.tool_calls.append({"name": call.name, "arguments": call.arguments})
                result = await session.call_tool(call.name, call.arguments)
                text = "".join(c.text for c in result.content if c.type == "text")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    except Exception as exc:  # keep the run; a dropped task would bias success upward
        run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
    return run


def run_tasks(
    tasks: Sequence[AgentTask],
    policy_for: Callable[[AgentTask], Policy],
    dataset_dir: Path = DEFAULT_DATASET,
    max_steps: int = 8,
) -> list[AgentRun]:
    """Run tasks sequentially against one MCP server subprocess."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "maxionbench.agents.mcp_server", "--dataset", str(dataset_dir)]
    )

    async def main() -> list[AgentRun]:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return [await run_agent(session, policy_for(t), t, max_steps) for t in tasks]

    return asyncio.run(main())


class ChatPolicy:
    """A chat model as the policy: tool calls in its reply are executed, plain content is the answer.

    `send(messages, tools)` returns a `CompletionResult` whose text is the assistant message JSON
    (see `maxionbench.eval.tool_client`). Per-step results are kept for latency and cost accounting.
    """

    def __init__(self, send: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]) -> None:
        self.send = send
        self.results: list[Any] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> PolicyOutput:
        result = self.send(messages, tools)
        self.results.append(result)
        if result.status != "ok":
            raise RuntimeError(f"model call failed: {result.error}")
        message = json.loads(result.text)
        calls = message.get("tool_calls") or []
        if not calls:
            return PolicyOutput(answer=(message.get("content") or "").strip())
        parsed = []
        for n, c in enumerate(calls):
            c["id"] = c.get("id") or f"call{len(self.results)}-{n}"  # some engines omit ids; tool replies need them
            args = c["function"].get("arguments") or "{}"
            parsed.append(ToolCall(c["id"], c["function"]["name"], json.loads(args) if isinstance(args, str) else dict(args)))
        message.setdefault("content", None)
        return PolicyOutput(tool_calls=tuple(parsed), assistant_message=message)


class OraclePolicy:
    """Search for each gold paragraph by its opening words, read it, then answer if the reads support it."""

    def __init__(self, task: AgentTask, corpus: HotpotCorpus) -> None:
        self.task = task
        self.queries = [" ".join(corpus.texts[d].split()[:12]) for d in task.gold_doc_ids]
        self.step = 0
        self.read_texts: list[str] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> PolicyOutput:
        last = messages[-1]["content"] if messages[-1]["role"] == "tool" else ""
        doc_i, phase = divmod(self.step, 2)
        self.step += 1
        if phase == 1:  # last message holds search results for gold doc `doc_i`
            want = self.task.gold_doc_ids[doc_i]
            if want not in re.findall(r"^\[([^\]]+)\]", last, flags=re.M):
                return PolicyOutput(answer="unknown")
            return PolicyOutput(tool_calls=(ToolCall(f"c{self.step}", "read", {"doc_id": want}),))
        if doc_i > 0:
            self.read_texts.append(last)
        if doc_i < len(self.queries):
            return PolicyOutput(tool_calls=(ToolCall(f"c{self.step}", "search", {"query": self.queries[doc_i]}),))
        answer = self.task.answer
        norm = normalize_answer(answer)
        supported = norm in ("yes", "no") or any(norm in normalize_answer(t) for t in self.read_texts)
        return PolicyOutput(answer=answer if supported else "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate agentic HotpotQA tasks with the scripted oracle agent")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    corpus = HotpotCorpus(args.dataset)
    tasks = build_tasks(corpus, args.tasks, args.seed, dataset_dir=args.dataset)
    runs = run_tasks(tasks, lambda t: OraclePolicy(t, corpus), args.dataset)
    solved = sum(r.answer == t.answer for r, t in zip(runs, tasks))
    calls = [len(r.tool_calls) for r in runs]
    recall = sum(t.question_search_recall for t in tasks) / len(tasks)
    print(json.dumps({"tasks": len(tasks), "oracle_solved": solved, "mean_tool_calls": sum(calls) / len(calls),
                      "min_tool_calls": min(calls), "mean_question_search_recall": round(recall, 3)}))
    return 0 if solved == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
