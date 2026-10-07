"""Task-suite trial: does agentic BrowseComp-Plus (per-query corpus) suit context-policy experiments?

Two gates: the baseline agent (full history kept) succeeds on 40-70% of tasks, so policies can move
success either way, and contexts grow past 30k tokens, so context management matters. Gemini runs the
agent; the calibrated Gemini judge grades answers. A spend budget stops new tasks once reached.
Outputs hold ids and numbers only: BrowseComp-Plus text must not be published.

    python -m maxionbench.eval.agent_trial --n 5 --budget-usd 0.3    # pilot
"""

from __future__ import annotations

from argparse import ArgumentParser
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from maxionbench.agents.browsecomp_env import SYSTEM_PROMPT, BrowseTask, load_tasks, write_docs
from maxionbench.agents.loop import AgentRun, ChatPolicy, run_agent
from maxionbench.eval.batch import Call, bound_send, metered, retryable, run_calls
from maxionbench.eval.tool_client import chat_tools
from maxionbench.graders.judge import RUBRIC_VERSION, judge_messages, parse_label
from maxionbench.harness.budget import ModelPrice, cost_usd
from maxionbench.harness.results import mean_ci
from maxionbench.harness.targets import GeminiTarget

MODEL = {"model": "gemini-3.5-flash-lite", "reasoning_effort": "minimal"}
JUDGE = {"model": "gemini-3.5-flash-lite", "reasoning_effort": "low"}
MAX_TOKENS = 512
LONG_CONTEXT = 30_000
SUCCESS_BAND = (0.40, 0.70)


def _log(msg: str) -> None:
    print(f"[agent-trial] {msg}", file=sys.stderr, flush=True)


def run_task(task: BrowseTask, policy: ChatPolicy, max_steps: int, workdir: Path) -> AgentRun:
    """One agent run against an MCP server holding only this task's documents."""
    docs = workdir / f"{task.task_id}.json"
    write_docs(task, docs)
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "maxionbench.agents.browsecomp_env", "--docs", str(docs)])

    async def main() -> AgentRun:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return await run_agent(session, policy, task, max_steps, SYSTEM_PROMPT)

    try:
        return asyncio.run(main())
    finally:
        docs.unlink()


def run_trial(n: int, max_steps: int, seed: int, pool: int, budget_usd: float, out_root: Path) -> dict[str, Any]:
    tasks = load_tasks(n, seed, pool=pool)
    _log(f"{len(tasks)} tasks, {statistics.median(len(t.docs) for t in tasks):.0f} docs each (median)")
    target, judge = GeminiTarget(MODEL), GeminiTarget(JUDGE)
    fn = bound_send(target, chat_tools)

    def estimate(price: ModelPrice) -> float:  # the budget plus one task with every step at 60k uncached tokens
        return budget_usd + max_steps * cost_usd(price, input_tokens=60_000, output_tokens=MAX_TOKENS)

    runs: list[tuple[BrowseTask, AgentRun, ChatPolicy]] = []
    with target, metered(target, estimate, "agent-trial/browsecomp") as meter, \
            tempfile.TemporaryDirectory() as tmp:
        def send(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
            for attempt in range(3):
                r = fn(target.base_urls[0], messages, tools=tools, max_tokens=MAX_TOKENS, timeout_s=120.0)
                meter.add(r, messages, MAX_TOKENS)
                if r.status == "ok" or not retryable(r.error):
                    return r
                time.sleep(2.0 * (attempt + 1))
            return r

        for i, task in enumerate(tasks):
            if meter.spend_usd >= budget_usd:
                _log(f"budget ${budget_usd:.2f} reached after {i} tasks")
                break
            policy = ChatPolicy(send)
            runs.append((task, run_task(task, policy, max_steps, Path(tmp)), policy))
            _log(f"task {i + 1}/{len(tasks)}: {runs[-1][1].status}, {len(policy.results)} model calls, "
                 f"spend ${meter.spend_usd:.4f}")
    agent_spend = meter.spend_usd

    answered = [(t, r) for t, r, _ in runs if r.status == "answered" and r.answer]
    calls = [Call(t.task_id, judge_messages(t.question, [t.answer], r.answer or ""), max_tokens=512) for t, r in answered]
    with judge:
        verdicts, judge_usage = run_calls(judge, calls, "agent-trial/judge", workers=8) if calls else ({}, {"spend_usd": 0.0})

    items = []
    for task, run, policy in runs:
        steps = [r for r in policy.results if r.status == "ok"]
        label = parse_label(verdicts[task.task_id].text) if task.task_id in verdicts else None
        items.append({
            "task_id": task.task_id,
            "run_status": run.status,
            "judge_label": label or "unjudged",
            "correct": label == "correct",
            "docs": len(task.docs),
            "model_calls": len(policy.results),
            "tool_calls": [c["name"] for c in run.tool_calls],
            "read_gold": sum(c["name"] == "read" and c["arguments"].get("doc_id") in task.gold_doc_ids
                             for c in run.tool_calls),
            "peak_context_tokens": max((r.prompt_tokens for r in steps), default=0),
            "prompt_tokens": sum(r.prompt_tokens for r in steps),
            "cached_tokens": sum(r.cached_tokens for r in steps),
            "error_type": (run.error or "").split(":", 1)[0] or None,  # type only: messages may quote task text
        })
    summary = summarize(items)
    summary.update({"n_planned": n, "max_steps": max_steps, "seed": seed, "pool": pool, "model": MODEL, "judge": JUDGE,
                    "judge_rubric": RUBRIC_VERSION, "spend_usd": {"agent": round(agent_spend, 4),
                                                                  "judge": judge_usage["spend_usd"]}})
    out_dir = out_root / f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-browsecomp-trial"
    out_dir.mkdir(parents=True)
    (out_dir / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote {out_dir}")
    return summary


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"tasks": 0}
    success = mean_ci([float(i["correct"]) for i in items])
    peaks = [i["peak_context_tokens"] for i in items]
    long_share = sum(p > LONG_CONTEXT for p in peaks) / len(items)
    return {
        "tasks": len(items),
        "success": {"mean": success.mean, "ci95": [success.ci_low, success.ci_high]},
        "answered_rate": sum(i["run_status"] == "answered" for i in items) / len(items),
        "peak_context_tokens": {"median": statistics.median(peaks), "max": max(peaks)},
        "long_context_share": long_share,
        "cached_share": sum(i["cached_tokens"] for i in items) / max(1, sum(i["prompt_tokens"] for i in items)),
        "model_calls_mean": statistics.fmean(i["model_calls"] for i in items),
        "passes": {"success_band": SUCCESS_BAND[0] <= success.mean <= SUCCESS_BAND[1],
                   "long_context": long_share >= 0.5},
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="BrowseComp-Plus task-suite trial on Gemini")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=1, help="queries whose documents each task searches")
    parser.add_argument("--budget-usd", type=float, required=True, help="stop starting tasks past this agent spend")
    parser.add_argument("--out", type=Path, default=Path("artifacts/agent_trial"))
    args = parser.parse_args(argv)
    print(json.dumps(run_trial(args.n, args.max_steps, args.seed, args.pool, args.budget_usd, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
