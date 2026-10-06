"""Loaders for the v0.3 public datasets (files pinned in manifests/v03.yaml).

Each loader takes its input via `sources.verified_path`, so a missing or modified file fails
loudly instead of silently changing results. Pass `path=` to load an unpinned file (tests).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from maxionbench.datasets.sources import verified_path

CRAG_SLICE = "crag/crag_task_1_and_2_dev_v4.first_500.jsonl"
SHAREGPT_SAMPLE = "sharegpt/sharegpt_first_turn.sample_2000.jsonl"
AZURE_CONV_TRACE = "azure/AzureLLMInferenceTrace_conv_1week.csv"
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


@dataclass(frozen=True)
class CragExample:
    interaction_id: str
    query: str
    query_time: str
    answer: str
    alt_ans: tuple[str, ...]
    domain: str
    question_type: str
    static_or_dynamic: str
    pages: tuple[dict[str, str], ...]  # page_name, page_url, page_snippet, page_last_modified


def load_crag(path: Path | None = None) -> list[CragExample]:
    """CRAG-500: the first 500 task 1&2 dev records (page HTML dropped)."""
    rows = _jsonl(path or verified_path(CRAG_SLICE))
    return [
        CragExample(
            interaction_id=r["interaction_id"],
            query=r["query"],
            query_time=r.get("query_time", ""),
            answer=str(r["answer"]),
            alt_ans=tuple(str(a) for a in r.get("alt_ans") or ()),
            domain=r.get("domain", ""),
            question_type=r.get("question_type", ""),
            static_or_dynamic=r.get("static_or_dynamic", ""),
            pages=tuple(r.get("search_results") or ()),
        )
        for r in rows
    ]


@dataclass(frozen=True)
class BeirDataset:
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title", "text"}
    queries: dict[str, str]  # query_id -> text (only queries judged in `split`)
    qrels: dict[str, dict[str, int]]  # query_id -> doc_id -> relevance


def load_beir(subset: str, split: str = "test", path: Path | None = None) -> BeirDataset:
    """A BEIR subset (scifact, fiqa) read straight from its pinned zip."""
    with zipfile.ZipFile(path or verified_path(f"beir/{subset}.zip")) as zf:
        def member(name: str) -> io.TextIOWrapper:
            return io.TextIOWrapper(zf.open(f"{subset}/{name}"), encoding="utf-8")

        qrels: dict[str, dict[str, int]] = {}
        with member(f"qrels/{split}.tsv") as fh:
            next(fh)  # header
            for line in fh:
                qid, doc_id, score = line.rstrip("\n").split("\t")
                qrels.setdefault(qid, {})[doc_id] = int(score)
        with member("queries.jsonl") as fh:
            queries = {r["_id"]: r["text"] for r in map(json.loads, fh) if r["_id"] in qrels}
        with member("corpus.jsonl") as fh:
            corpus = {r["_id"]: {"title": r.get("title", ""), "text": r["text"]} for r in map(json.loads, fh)}
    return BeirDataset(corpus=corpus, queries=queries, qrels=qrels)


@dataclass(frozen=True)
class ChatPair:
    id: str
    prompt: str
    completion: str


def load_sharegpt(path: Path | None = None) -> list[ChatPair]:
    """2,000 seeded first-turn ShareGPT pairs (the vLLM serving-benchmark convention)."""
    return [ChatPair(r["id"], r["prompt"], r["completion"]) for r in _jsonl(path or verified_path(SHAREGPT_SAMPLE))]


@dataclass(frozen=True)
class TraceWindow:
    arrival_s: np.ndarray  # seconds since window start, non-decreasing
    context_tokens: np.ndarray
    generated_tokens: np.ndarray
    duration_s: float

    @property
    def rate_rps(self) -> float:
        return len(self.arrival_s) / self.duration_s


@functools.cache
def load_azure_trace(start_s: float, duration_s: float, path: Path | None = None) -> TraceWindow:
    """Requests whose arrival lies in [start_s, start_s + duration_s) after the trace's first request.

    Reads in chunks and stops past the window, so short windows near the start are cheap.
    """
    if duration_s <= 0 or start_s < 0:
        raise ValueError("need start_s >= 0 and duration_s > 0")
    src = path or verified_path(AZURE_CONV_TRACE)
    parts = []
    t0 = None
    for chunk in pd.read_csv(src, chunksize=500_000):
        ts = pd.to_datetime(chunk["TIMESTAMP"], format="ISO8601")
        if t0 is None:
            t0 = ts.iloc[0]
        offset = (ts - t0).dt.total_seconds().to_numpy()
        keep = (offset >= start_s) & (offset < start_s + duration_s)
        parts.append((offset[keep] - start_s, chunk["ContextTokens"].to_numpy()[keep],
                      chunk["GeneratedTokens"].to_numpy()[keep]))
        if offset[-1] >= start_s + duration_s:
            break
    arrival, ctx, gen = (np.concatenate(cols) for cols in zip(*parts))
    if not len(arrival):
        raise ValueError(f"no requests in trace window [{start_s}, {start_s + duration_s})")
    order = np.argsort(arrival, kind="stable")
    return TraceWindow(arrival[order], ctx[order].astype(int), gen[order].astype(int), float(duration_s))


@dataclass(frozen=True)
class BfclCase:
    id: str
    category: str
    messages: tuple[dict[str, str], ...]
    functions: tuple[dict[str, Any], ...]  # BFCL function docs (JSON-schema-like, "dict" for object)
    ground_truth: tuple[dict[str, dict[str, list[Any]]], ...]  # [{func_name: {param: [acceptable values]}}]


def load_bfcl(category: str, root: Path | None = None) -> list[BfclCase]:
    """BFCL v3 single-turn AST categories with their possible answers."""
    if category not in BFCL_CATEGORIES:
        raise ValueError(f"category must be one of {BFCL_CATEGORIES}")
    name = f"BFCL_v3_{category}.json"
    if root is None:
        questions, answers = verified_path(f"bfcl/{name}"), verified_path(f"bfcl/possible_answer/{name}")
    else:
        questions, answers = root / name, root / "possible_answer" / name
    truth = {r["id"]: r["ground_truth"] for r in _jsonl(answers)}
    return [
        BfclCase(
            id=r["id"],
            category=category,
            messages=tuple(r["question"][0]),  # single-turn: one turn of messages
            functions=tuple(r["function"]),
            ground_truth=tuple(truth[r["id"]]),
        )
        for r in _jsonl(questions)
    ]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
