from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from maxionbench.datasets.loaders.processed import (
    PROCESSED_SCHEMA_VERSION,
    load_processed_ann_dataset,
    load_processed_d4_bundle,
    load_processed_filtered_ann_dataset,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for qid, did, rel in rows:
            handle.write(f"{qid}\t{did}\t{rel}\n")


def test_load_processed_ann_dataset_reads_numeric_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "D1" / "glove-100-angular"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "ann",
            "metric": "angular",
        },
    )
    np.save(root / "base.npy", np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32))
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[0, 2]], dtype=np.int32))

    ds = load_processed_ann_dataset(root, top_k=2)
    assert ds.metric == "angular"
    assert ds.ids == ["doc-0000000", "doc-0000001", "doc-0000002"]
    assert ds.ground_truth_ids == [["doc-0000000", "doc-0000002"]]


def test_load_processed_ann_dataset_requires_explicit_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "D1" / "glove-100-angular"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "task_type": "ann",
            "metric": "angular",
        },
    )
    np.save(root / "base.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[0]], dtype=np.int32))

    with pytest.raises(ValueError, match="must declare schema_version"):
        load_processed_ann_dataset(root, top_k=1)


def test_load_processed_ann_dataset_rejects_out_of_bounds_gt_ids(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "processed" / "D1" / "glove-100-angular"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "ann",
            "metric": "angular",
        },
    )
    np.save(root / "base.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[7]], dtype=np.int32))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="out-of-bounds"):
            load_processed_ann_dataset(root, top_k=1)
    assert "out-of-bounds ground-truth indices" in caplog.text


def test_load_processed_filtered_ann_dataset_reads_filters_and_payloads(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "D3" / "yfcc-10M"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "filtered_ann",
            "metric": "l2",
        },
    )
    np.save(root / "base.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[0, 1]], dtype=np.int32))
    _write_jsonl(root / "filters.jsonl", [{"query_id": 0, "must_have_tags": ["tag-a"]}])
    _write_jsonl(root / "payloads.jsonl", [{"tags": ["tag-a"]}, {"tags": ["tag-b"]}])

    ds = load_processed_filtered_ann_dataset(root, top_k=2)
    assert ds.query_filters == [{"query_id": 0, "must_have_tags": ["tag-a"]}]
    assert ds.payloads[0]["tags"] == ["tag-a"]
    assert ds.ground_truth_ids[0] == ["doc-0000000", "doc-0000001"]


def test_load_processed_filtered_ann_dataset_warns_on_payload_truncation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "processed" / "D3" / "yfcc-10M"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "filtered_ann",
            "metric": "l2",
        },
    )
    np.save(root / "base.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "queries.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(root / "gt_ids.npy", np.asarray([[0, 1]], dtype=np.int32))
    _write_jsonl(root / "filters.jsonl", [{"query_id": 0, "must_have_tags": ["tag-a"]}])
    _write_jsonl(root / "payloads.jsonl", [{"idx": 0}, {"idx": 1}, {"idx": 2}])

    with caplog.at_level(logging.WARNING):
        ds = load_processed_filtered_ann_dataset(root, top_k=2)
    assert len(ds.payloads) == 2
    assert "truncating extras" in caplog.text


def test_load_processed_d4_bundle_merges_beir_and_crag(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "D4"

    beir = root / "beir" / "fiqa"
    beir.mkdir(parents=True)
    _write_json(
        beir / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        beir / "corpus.jsonl",
        [
            {"doc_id": "fiqa::doc::d1", "title": "Bond", "text": "bond market"},
            {"doc_id": "fiqa::doc::d2", "title": "Genome", "text": "genome medicine"},
        ],
    )
    _write_jsonl(beir / "queries.jsonl", [{"query_id": "fiqa::q::q1", "text": "bond market"}])
    _write_qrels(beir / "qrels.tsv", [("fiqa::q::q1", "fiqa::doc::d1", 2)])

    crag = root / "crag" / "small_slice"
    crag.mkdir(parents=True)
    _write_json(
        crag / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_weak",
        },
    )
    _write_jsonl(
        crag / "corpus.jsonl",
        [{"doc_id": "crag_small_slice::doc::1_p0_c0", "title": "Weather", "text": "weather report"}],
    )
    _write_jsonl(
        crag / "queries.jsonl",
        [{"query_id": "crag_small_slice::q::1", "text": "weather report"}],
    )
    _write_qrels(crag / "qrels.tsv", [("crag_small_slice::q::1", "crag_small_slice::doc::1_p0_c0", 1)])

    ds = load_processed_d4_bundle(
        root,
        vector_dim=8,
        seed=5,
        beir_subsets=["fiqa"],
        include_crag=True,
        max_docs=10,
        max_queries=10,
    )
    assert len(ds.doc_ids) == 3
    assert len(ds.query_ids) == 2
    assert any(doc_id.startswith("fiqa::doc::") for doc_id in ds.doc_ids)
    assert any(doc_id.startswith("crag_small_slice::doc::") for doc_id in ds.doc_ids)


def test_load_processed_d4_bundle_logs_query_drop_warnings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "processed" / "D4"

    beir = root / "beir" / "fiqa"
    beir.mkdir(parents=True)
    _write_json(
        beir / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        beir / "corpus.jsonl",
        [{"doc_id": "fiqa::doc::d1", "title": "Bond", "text": "bond market"}],
    )
    _write_jsonl(
        beir / "queries.jsonl",
        [
            {"query_id": "fiqa::q::q1", "text": "bond market"},
            {"query_id": "fiqa::q::q2", "text": "dropped query"},
        ],
    )
    _write_qrels(beir / "qrels.tsv", [("fiqa::q::q1", "fiqa::doc::d1", 2)])

    crag = root / "crag" / "small_slice"
    crag.mkdir(parents=True)
    _write_json(
        crag / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_weak",
        },
    )
    _write_jsonl(
        crag / "corpus.jsonl",
        [{"doc_id": "crag_small_slice::doc::1", "title": "Weather", "text": "weather report"}],
    )
    _write_jsonl(
        crag / "queries.jsonl",
        [{"query_id": "crag_small_slice::q::1", "text": "weather report"}],
    )
    _write_qrels(crag / "qrels.tsv", [("crag_small_slice::q::1", "crag_small_slice::doc::1", 1)])

    with caplog.at_level(logging.WARNING):
        ds = load_processed_d4_bundle(
            root,
            vector_dim=8,
            seed=5,
            beir_subsets=["fiqa"],
            include_crag=True,
            max_docs=1,
            max_queries=10,
        )
    assert len(ds.query_ids) >= 1
    assert "dropped 1 queries without surviving qrels" in caplog.text
    assert "processed D4 merge dropped docs/queries during bundle merge" in caplog.text


def test_load_processed_d4_bundle_raises_when_all_queries_drop(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "processed" / "D4"

    beir = root / "beir" / "fiqa"
    beir.mkdir(parents=True)
    _write_json(
        beir / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        beir / "corpus.jsonl",
        [{"doc_id": "fiqa::doc::d1", "title": "Bond", "text": "bond market"}],
    )
    _write_jsonl(
        beir / "queries.jsonl",
        [{"query_id": "fiqa::q::q1", "text": "bond market"}],
    )
    _write_qrels(beir / "qrels.tsv", [("fiqa::q::q1", "fiqa::doc::missing", 2)])

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="processed D4 merge produced 0 queries after filtering"):
            load_processed_d4_bundle(
                root,
                vector_dim=8,
                seed=5,
                beir_subsets=["fiqa"],
                include_crag=False,
                max_queries=10,
            )
    assert "dropped 1 queries without surviving qrels" in caplog.text


def test_load_processed_d4_bundle_accepts_qrels_header_row(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "D4"
    beir = root / "beir" / "fiqa"
    beir.mkdir(parents=True)
    _write_json(
        beir / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        beir / "corpus.jsonl",
        [{"doc_id": "fiqa::doc::d1", "title": "Bond", "text": "bond market"}],
    )
    _write_jsonl(
        beir / "queries.jsonl",
        [{"query_id": "fiqa::q::q1", "text": "bond market"}],
    )
    (beir / "qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\nfiqa::q::q1\tfiqa::doc::d1\t2\n",
        encoding="utf-8",
    )

    ds = load_processed_d4_bundle(
        root,
        vector_dim=8,
        seed=5,
        beir_subsets=["fiqa"],
        include_crag=False,
        max_queries=10,
    )
    assert ds.query_ids == ["fiqa::q::q1"]
