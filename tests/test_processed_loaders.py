from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from maxionbench.datasets.loaders.processed import (
    PROCESSED_SCHEMA_VERSION,
    embedding_model_slug,
    load_processed_ann_dataset,
    load_processed_d4_bundle,
    load_processed_filtered_ann_dataset,
    load_processed_text_dataset,
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


def _write_precomputed_embeddings(
    dataset_dir: Path,
    *,
    model_id: str,
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
) -> None:
    embedding_dir = dataset_dir / "embeddings" / embedding_model_slug(model_id)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / "doc_vectors.npy", np.asarray(doc_vectors, dtype=np.float32))
    np.save(embedding_dir / "query_vectors.npy", np.asarray(query_vectors, dtype=np.float32))
    doc_digest = json.dumps(doc_ids, separators=(",", ":")).encode("utf-8")
    query_digest = json.dumps(query_ids, separators=(",", ":")).encode("utf-8")
    _write_json(
        embedding_dir / "meta.json",
        {
            "schema_version": "maxionbench-text-embeddings-v1",
            "model_id": model_id,
            "dim": int(doc_vectors.shape[1]),
            "doc_count": int(doc_vectors.shape[0]),
            "query_count": int(query_vectors.shape[0]),
            "doc_ids_sha256": hashlib.sha256(doc_digest).hexdigest(),
            "query_ids_sha256": hashlib.sha256(query_digest).hexdigest(),
        },
    )


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
    assert list(ds.ids) == ["doc-0000000", "doc-0000001", "doc-0000002"]
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


def test_load_processed_text_dataset_prefers_precomputed_embeddings(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "frames_portable"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(
        root / "corpus.jsonl",
        [
            {"doc_id": "doc-a", "text": "misleading lexical tokens"},
            {"doc_id": "doc-b", "text": "also misleading lexical tokens"},
        ],
    )
    _write_jsonl(root / "queries.jsonl", [{"query_id": "q-1", "text": "misleading lexical tokens"}])
    _write_qrels(root / "qrels.tsv", [("q-1", "doc-a", 1)])
    _write_precomputed_embeddings(
        root,
        model_id="BAAI/bge-small-en-v1.5",
        doc_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        doc_ids=["doc-a", "doc-b"],
        query_ids=["q-1"],
    )

    ds = load_processed_text_dataset(
        root,
        vector_dim=2,
        seed=17,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=2,
        require_precomputed_embeddings=True,
    )
    assert ds.doc_ids == ["doc-a", "doc-b"]
    assert ds.query_ids == ["q-1"]
    assert np.allclose(ds.doc_vectors, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.allclose(ds.query_vectors, np.asarray([[1.0, 0.0]], dtype=np.float32))


def test_load_processed_text_dataset_requires_precomputed_embeddings_when_requested(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "frames_portable"
    root.mkdir(parents=True)
    _write_json(
        root / "meta.json",
        {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "task_type": "text_retrieval_strict",
        },
    )
    _write_jsonl(root / "corpus.jsonl", [{"doc_id": "doc-a", "text": "alpha"}])
    _write_jsonl(root / "queries.jsonl", [{"query_id": "q-1", "text": "alpha"}])
    _write_qrels(root / "qrels.tsv", [("q-1", "doc-a", 1)])

    with pytest.raises(FileNotFoundError, match="missing precomputed embeddings"):
        load_processed_text_dataset(
            root,
            vector_dim=2,
            seed=17,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=2,
            require_precomputed_embeddings=True,
        )
