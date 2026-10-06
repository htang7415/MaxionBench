"""Agentic HotpotQA: BM25 search/read tools over the processed HotpotQA corpus, and the task set.

Tasks keep only questions a single search cannot answer: the question's own top-k results must
miss at least one gold paragraph, so an agent has to issue follow-up searches (multi-hop).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import bm25s

from maxionbench.rag.answer_metrics import normalize_answer

DEFAULT_DATASET = Path("dataset/processed/hotpot_portable")
_DOC_PREFIX = "hotpotqa_portable::doc::"


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    snippet: str


class HotpotCorpus:
    """Paragraph store with BM25 search. Doc ids are the short hash after the dataset prefix."""

    def __init__(self, dataset_dir: Path = DEFAULT_DATASET) -> None:
        self.texts: dict[str, str] = {}
        with (Path(dataset_dir) / "corpus.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                self.texts[short_id(row["doc_id"])] = row["text"]
        self._ids = list(self.texts)
        self._bm25 = bm25s.BM25()
        self._bm25.index(bm25s.tokenize(list(self.texts.values()), show_progress=False), show_progress=False)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        k = max(1, min(int(k), 20))
        tokens = bm25s.tokenize([query], show_progress=False)
        if not tokens.vocab:  # only stopwords: nothing to match
            return []
        idx, _ = self._bm25.retrieve(tokens, k=k, show_progress=False)
        return [SearchHit(self._ids[i], self.texts[self._ids[i]][:200]) for i in idx[0].tolist()]

    def read(self, doc_id: str) -> str | None:
        return self.texts.get(short_id(doc_id))


def short_id(doc_id: str) -> str:
    return doc_id.removeprefix(_DOC_PREFIX)


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    question: str
    answer: str
    gold_doc_ids: tuple[str, ...]
    question_search_recall: float  # share of gold paragraphs in the question's own top-k


def build_tasks(
    corpus: HotpotCorpus, n: int, seed: int = 0, k: int = 5, dataset_dir: Path = DEFAULT_DATASET
) -> list[AgentTask]:
    """`n` seeded multi-hop tasks whose answer is stated in the gold paragraphs (or is yes/no)."""
    gold: dict[str, list[str]] = {}
    with (Path(dataset_dir) / "qrels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            gold.setdefault(qid, []).append(short_id(doc_id))
    with (Path(dataset_dir) / "queries.jsonl").open(encoding="utf-8") as fh:
        queries = [r for r in map(json.loads, fh) if r["query_id"] in gold]
    random.Random(seed).shuffle(queries)
    tasks = []
    for q in queries:
        docs = gold[q["query_id"]]
        hits = {h.doc_id for h in corpus.search(q["text"], k)}
        recall = sum(d in hits for d in docs) / len(docs)
        if recall == 1.0 or not _answer_in_docs(q["answer"], [corpus.texts[d] for d in docs]):
            continue
        tasks.append(AgentTask(q["query_id"].rsplit("::", 1)[-1], q["text"], q["answer"], tuple(docs), recall))
        if len(tasks) == n:
            break
    return tasks


def _answer_in_docs(answer: str, texts: list[str]) -> bool:
    norm = normalize_answer(answer)
    return norm in ("yes", "no") or any(norm in normalize_answer(t) for t in texts)
