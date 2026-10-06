"""Short-answer RAG prompts for CRAG (web snippets) and HotpotQA (gold + distractor paragraphs)."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import random

from maxionbench.datasets.loaders.v03 import CragExample
from maxionbench.tools.rag_eval import build_messages

CRAG_SYSTEM = (
    "Answer the question using the web search results. Reply with the shortest possible answer and "
    "no explanation. If the results do not contain the answer, reply \"I don't know\". If the question "
    "rests on a false premise, reply \"invalid question\"."
)


@dataclass(frozen=True)
class QAItem:
    id: str
    source: str  # "crag" | "hotpotqa"
    question: str
    golds: tuple[str, ...]
    messages: tuple[dict[str, str], ...]


def crag_item(ex: CragExample, max_pages: int = 5, snippet_chars: int = 600) -> QAItem:
    pages = "\n\n".join(
        f"[{i}] {html.unescape(p.get('page_name') or '')}\n{html.unescape(p.get('page_snippet') or '')[:snippet_chars]}"
        for i, p in enumerate(ex.pages[:max_pages], start=1)
    )
    user = f"Search results:\n{pages}\n\nCurrent time: {ex.query_time}\nQuestion: {ex.query}"
    return QAItem(f"crag-{ex.interaction_id}", "crag", ex.query, (ex.answer, *ex.alt_ans),
                  ({"role": "system", "content": CRAG_SYSTEM}, {"role": "user", "content": user}))


def hotpot_items(dataset_dir: Path, n: int, seed: int, k: int = 5) -> list[QAItem]:
    """`n` seeded HotpotQA questions, each with its gold paragraphs plus random distractors (k total)."""
    rng = random.Random(seed)
    docs: dict[str, str] = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            docs[row["doc_id"]] = row["text"]
    gold: dict[str, list[str]] = {}
    with (dataset_dir / "qrels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            gold.setdefault(qid, []).append(doc_id)
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as fh:
        queries = [r for r in map(json.loads, fh) if r["query_id"] in gold]
    doc_ids = list(docs)
    items = []
    for q in rng.sample(queries, n):
        ctx = gold[q["query_id"]][:k]
        ctx += [d for d in rng.sample(doc_ids, k) if d not in ctx][: k - len(ctx)]
        rng.shuffle(ctx)
        items.append(QAItem(f"hotpot-{q['query_id'].rsplit('::', 1)[-1]}", "hotpotqa", q["text"], (q["answer"],),
                            tuple(build_messages(q["text"], [docs[d] for d in ctx]))))
    return items
