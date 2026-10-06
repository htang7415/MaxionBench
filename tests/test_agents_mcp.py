from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest

from maxionbench.agents.hotpot_env import HotpotCorpus, build_tasks
from maxionbench.agents.loop import OraclePolicy, PolicyOutput, ToolCall, run_tasks

P = "hotpotqa_portable::doc::"
DOCS = {
    "film": "Night Harbor is a 1961 drama film directed by Ottilie Brandt and set on a fishing boat.",
    "brandt": "Ottilie Brandt (1920-1988) was a German screenwriter raised in Lübeck.",
    "decoy": "Harbor City is a 1950 film made at a night studio; few people in the city saw it.",
    "other": "Cedar Falls is a small town known for its paper mill and annual river festival.",
    "lake": "Lake Varn is a glacial lake whose outflow powers the Cedar Falls paper mill.",
}
QUERIES = [  # (id, question, answer, gold docs)
    ("q1", "In which city did the person who made the film Night Harbor grow up?", "Lübeck", ["film", "brandt"]),
    ("q2", "Which lake powers the paper mill in Cedar Falls?", "Lake Varn", ["other", "lake"]),
]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hotpot")
    (root / "corpus.jsonl").write_text(
        "".join(json.dumps({"doc_id": P + k, "text": v}) + "\n" for k, v in DOCS.items()), encoding="utf-8"
    )
    (root / "queries.jsonl").write_text(
        "".join(json.dumps({"query_id": f"hotpotqa_portable::q::{q}", "text": t, "answer": a}) + "\n"
                for q, t, a, _ in QUERIES), encoding="utf-8")
    rows = ["query_id\tdoc_id\trelevance"]
    rows += [f"hotpotqa_portable::q::{q}\t{P}{d}\t1" for q, _, _, docs in QUERIES for d in docs]
    (root / "qrels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_tasks_keep_only_questions_one_search_cannot_answer(dataset: Path) -> None:
    corpus = HotpotCorpus(dataset)
    tasks = build_tasks(corpus, n=10, k=2, dataset_dir=dataset)
    # q2's own search returns both gold paragraphs; q1's second hop is out-ranked by the decoy
    assert [t.task_id for t in tasks] == ["q1"]
    assert tasks[0].gold_doc_ids == ("film", "brandt") and tasks[0].question_search_recall == 0.5
    assert build_tasks(corpus, n=10, k=5, dataset_dir=dataset) == []  # k covers the corpus: nothing is multi-hop


def test_mcp_server_lists_and_runs_tools(dataset: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "maxionbench.agents.mcp_server", "--dataset", str(dataset)]
    )

    async def session_calls() -> tuple[set[str], str, str, str]:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            hits = await session.call_tool("search", {"query": "Ottilie Brandt born", "k": 2})
            doc = await session.call_tool("read", {"doc_id": "brandt"})
            missing = await session.call_tool("read", {"doc_id": "nope"})
            return names, hits.content[0].text, doc.content[0].text, missing.content[0].text

    names, hits, doc, missing = asyncio.run(session_calls())
    assert names == {"search", "read"}
    assert hits.splitlines()[0].startswith("[brandt] Ottilie Brandt") and len(hits.splitlines()) == 2
    assert doc == DOCS["brandt"] and "unknown doc_id" in missing


def test_scripted_agent_solves_known_tasks_over_mcp(dataset: Path) -> None:
    corpus = HotpotCorpus(dataset)
    tasks = build_tasks(corpus, n=10, k=2, dataset_dir=dataset)
    runs = run_tasks(tasks, lambda t: OraclePolicy(t, corpus), dataset)
    assert [(r.status, r.answer) for r in runs] == [("answered", "Lübeck")]
    assert [c["name"] for c in runs[0].tool_calls] == ["search", "read", "search", "read"]


def test_agent_run_stops_at_max_steps(dataset: Path) -> None:
    corpus = HotpotCorpus(dataset)
    tasks = build_tasks(corpus, n=1, k=2, dataset_dir=dataset)

    def looping(messages, tools):
        return PolicyOutput(tool_calls=(ToolCall("c", "search", {"query": "film"}),))

    (run,) = run_tasks(tasks, lambda t: looping, dataset, max_steps=3)
    assert run.status == "max_steps" and run.answer is None and len(run.tool_calls) == 3
