from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from maxionbench.datasets.loaders.processed import (
    _ProcessedTextBundle,
    _merge_text_bundles,
    load_processed_text_dataset,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_processed_dataset(
    root: Path,
    *,
    docs: list[dict[str, str]],
    queries: list[dict[str, str]],
    qrels: list[tuple[str, str, int]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": "maxionbench-processed-v1",
                "task_type": "text_retrieval_strict",
                "name": root.name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(root / "corpus.jsonl", docs)
    _write_jsonl(root / "queries.jsonl", queries)
    with (root / "qrels.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in qrels:
            handle.write(f"{qid}\t{did}\t{score}\n")


def test_merge_text_bundles_prioritizes_qrel_docs_under_cap() -> None:
    bundle = _ProcessedTextBundle(
        doc_ids=["filler-1", "filler-2", "evidence-1"],
        doc_texts=["f1", "f2", "e1"],
        query_ids=["q1"],
        query_texts=["query"],
        qrels={"q1": {"evidence-1": 1}},
    )

    merged = _merge_text_bundles(
        bundles=[bundle],
        max_docs=2,
        max_queries=10,
        prioritize_qrel_docs=True,
    )

    assert merged.doc_ids == ["evidence-1", "filler-1"]
    assert merged.query_ids == ["q1"]
    assert merged.qrels["q1"] == {"evidence-1": 1}


def test_merge_text_bundles_reports_truthful_drops_when_priority_docs_exceed_cap(caplog: pytest.LogCaptureFixture) -> None:
    bundle = _ProcessedTextBundle(
        doc_ids=["evidence-1", "evidence-2", "evidence-3"],
        doc_texts=["e1", "e2", "e3"],
        query_ids=["q1"],
        query_texts=["query"],
        qrels={"q1": {"evidence-1": 1, "evidence-2": 1, "evidence-3": 1}},
    )

    with caplog.at_level(logging.WARNING):
        merged = _merge_text_bundles(
            bundles=[bundle],
            max_docs=2,
            max_queries=10,
            prioritize_qrel_docs=True,
        )

    assert merged.doc_ids == ["evidence-1", "evidence-2"]
    assert "doc_limit_drops=1" in caplog.text
    assert "query_missing_qrels=0" in caplog.text


def test_load_processed_text_dataset_sparse_query_guard_uses_raw_query_count(tmp_path: Path) -> None:
    dataset_root = tmp_path / "events"
    _write_processed_dataset(
        dataset_root,
        docs=[
            {"doc_id": "d1", "text": "doc 1"},
            {"doc_id": "d2", "text": "doc 2"},
            {"doc_id": "d3", "text": "doc 3"},
            {"doc_id": "d4", "text": "doc 4"},
            {"doc_id": "d5", "text": "doc 5"},
            {"doc_id": "d6", "text": "doc 6"},
            {"doc_id": "d7", "text": "doc 7"},
            {"doc_id": "d8", "text": "doc 8"},
            {"doc_id": "d9", "text": "doc 9"},
            {"doc_id": "d10", "text": "doc 10"},
        ],
        queries=[{"query_id": f"q{i}", "text": f"query {i}"} for i in range(1, 11)],
        qrels=[(f"q{i}", f"d{i}", 1) for i in range(1, 11)],
    )

    with pytest.raises(RuntimeError, match=r"5/10 usable queries retained"):
        load_processed_text_dataset(
            dataset_root,
            vector_dim=8,
            seed=7,
            max_docs=5,
            max_queries=5000,
            prioritize_qrel_docs=False,
            min_query_retention_ratio=0.9,
        )
