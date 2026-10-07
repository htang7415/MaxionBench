from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from maxionbench.agents import browsecomp_env
from maxionbench.agents.browsecomp_env import CANARY, SHARDS, BrowseTask, DocCorpus, decrypt, load_tasks
from maxionbench.agents.loop import PolicyOutput, ToolCall
from maxionbench.eval.agent_trial import run_task, summarize


def encrypt(text: str) -> str:  # upstream's obfuscation: XOR with the repeated SHA-256 of the canary
    data = text.encode("utf-8")
    digest = hashlib.sha256(CANARY.encode("utf-8")).digest()
    key = digest * (len(data) // len(digest)) + digest[: len(data) % len(digest)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode("ascii")


def _docs(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"docid": encrypt(d), "text": encrypt(t), "url": encrypt("https://example.org/" + d)} for d, t in pairs]


@pytest.fixture
def shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(browsecomp_env, "verified_path", lambda rel, root: Path(root) / rel)
    for i, rel in enumerate(SHARDS):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "query_id": str(10 * i + j),
            "query": encrypt(f"question {i}-{j} über"),
            "answer": encrypt(f"answer {i}-{j}"),
            "gold_docs": _docs((f"g{i}{j}", "gold page")),
            "evidence_docs": _docs((f"g{i}{j}", "gold page"), (f"e{i}{j}", "evidence page")),
            "negative_docs": _docs((f"n{i}{j}", "negative page")),
        } for j in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), path)
    return tmp_path


def test_decrypt_inverts_upstream_obfuscation() -> None:
    text = "Ottilie Brandt grew up in Lübeck. " * 5  # longer than one 32-byte key block, non-ASCII
    assert decrypt(encrypt(text)) == text


def test_load_tasks_decrypts_seeded_sample_with_its_own_documents(shards: Path) -> None:
    tasks = load_tasks(4, seed=1, root=shards)
    assert [t.task_id for t in tasks] == [t.task_id for t in load_tasks(4, seed=1, root=shards)]
    assert len({t.task_id for t in tasks}) == 4
    for t in tasks:
        i, j = divmod(int(t.task_id), 10)
        assert (t.question, t.answer) == (f"question {i}-{j} über", f"answer {i}-{j}")
        assert t.docs == {f"g{i}{j}": "gold page", f"e{i}{j}": "evidence page", f"n{i}{j}": "negative page"}
        assert t.gold_doc_ids == (f"g{i}{j}",)
    pooled = load_tasks(4, seed=1, root=shards, pool=2)
    assert [t.task_id for t in pooled] == [t.task_id for t in tasks]  # same tasks for every pool size
    for own, t in zip(tasks, pooled):
        assert own.docs.items() <= t.docs.items() and len(t.docs) == 6  # plus one other query's three documents


def test_doc_corpus_ranks_matches_and_caps_reads() -> None:
    corpus = DocCorpus({"a": "the harbor film by Brandt", "b": "paper mill on the river", "c": "x" * 50_000})
    assert corpus.search("Brandt film", k=1)[0].doc_id == "a"
    assert corpus.search("the", k=3) == []  # stopwords only
    assert len(corpus.read("c") or "") == browsecomp_env.READ_CHARS
    assert corpus.read("missing") is None


class ScriptedPolicy:
    """Search, read the top hit, then answer with the first word of the page."""

    def __init__(self) -> None:
        self.results: list[Any] = []
        self.step = 0

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> PolicyOutput:
        self.step += 1
        last = messages[-1]["content"]
        if self.step == 1:
            return PolicyOutput(tool_calls=(ToolCall("c1", "search", {"query": "mill river"}),))
        if self.step == 2:
            doc_id = re.findall(r"^\[([^\]]+)\]", last, flags=re.M)[0]
            return PolicyOutput(tool_calls=(ToolCall("c2", "read", {"doc_id": doc_id}),))
        return PolicyOutput(answer=last.split()[0])


def test_run_task_serves_only_the_tasks_documents_over_mcp(tmp_path: Path) -> None:
    task = BrowseTask("7", "Which mill?", "paper", {"m": "paper mill on the river", "h": "harbor film"}, ("m",))
    run = run_task(task, ScriptedPolicy(), max_steps=5, workdir=tmp_path)  # type: ignore[arg-type]
    assert (run.status, run.answer) == ("answered", "paper")
    assert [c["name"] for c in run.tool_calls] == ["search", "read"]
    assert list(tmp_path.iterdir()) == []  # the decrypted document file is removed


def test_summarize_applies_both_gates() -> None:
    def item(correct: bool, peak: int) -> dict[str, Any]:
        return {"correct": correct, "run_status": "answered", "peak_context_tokens": peak, "cached_tokens": 50,
                "prompt_tokens": 100, "model_calls": 4}

    fits = summarize([item(True, 40_000), item(False, 35_000), item(True, 10_000), item(False, 31_000)])
    assert fits["success"]["mean"] == 0.5 and fits["long_context_share"] == 0.75
    assert fits["passes"] == {"success_band": True, "long_context": True}
    assert summarize([item(False, 5_000)] * 3)["passes"] == {"success_band": False, "long_context": False}
