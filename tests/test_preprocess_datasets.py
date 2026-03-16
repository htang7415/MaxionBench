from __future__ import annotations

import bz2
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest

from maxionbench.tools.preprocess_datasets import (
    preprocess_ann_hdf5,
    preprocess_beir_dataset,
    preprocess_crag_small_slice,
    preprocess_d3_from_explicit_files,
    preprocess_d3_yfcc_raw,
    preprocess_d3_yfcc_raw_to_explicit,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_u8bin(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.uint8)
    assert arr.ndim == 2
    with path.open("wb") as handle:
        np.asarray(arr.shape, dtype=np.uint32).tofile(handle)
        arr.tofile(handle)


def _write_ibin(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.int32)
    assert arr.ndim == 2
    with path.open("wb") as handle:
        np.asarray(arr.shape, dtype=np.int32).tofile(handle)
        arr.tofile(handle)


def _write_spmat(path: Path, rows: list[list[int]], *, ncol: int) -> None:
    nnz = sum(len(row) for row in rows)
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    with path.open("wb") as handle:
        np.asarray([len(rows), ncol, nnz], dtype=np.int64).tofile(handle)
        np.asarray(indptr, dtype=np.int64).tofile(handle)
        np.asarray(indices, dtype=np.int32).tofile(handle)
        np.ones(nnz, dtype=np.float32).tofile(handle)


def test_preprocess_ann_hdf5_writes_canonical_files(tmp_path: Path) -> None:
    source = tmp_path / "glove.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("train", data=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        handle.create_dataset("test", data=np.asarray([[1.0, 0.0]], dtype=np.float32))
        handle.create_dataset("neighbors", data=np.asarray([[0, 1]], dtype=np.int32))
        handle.attrs["distance"] = "angular"

    out_dir = tmp_path / "processed" / "D1" / "glove-100-angular"
    summary = preprocess_ann_hdf5(
        input_path=source,
        out_dir=out_dir,
        family="D1",
        dataset_name="glove-100-angular",
        metric="angular",
    )
    assert summary["dataset_name"] == "glove-100-angular"
    assert (out_dir / "base.npy").exists()
    assert (out_dir / "queries.npy").exists()
    assert (out_dir / "gt_ids.npy").exists()
    meta = _read_json(out_dir / "meta.json")
    assert meta["task_type"] == "ann"
    assert meta["family"] == "D1"


def test_preprocess_d3_explicit_writes_filters_and_payloads(tmp_path: Path) -> None:
    base = tmp_path / "base.npy"
    queries = tmp_path / "queries.npy"
    gt = tmp_path / "gt.npy"
    filters = tmp_path / "filters.jsonl"
    payloads = tmp_path / "payloads.jsonl"

    np.save(base, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(queries, np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(gt, np.asarray([[0, 1]], dtype=np.int32))
    filters.write_text(json.dumps({"query_id": 0, "must_have_tags": ["tag-a"]}) + "\n", encoding="utf-8")
    payloads.write_text(json.dumps({"tags": ["tag-a"]}) + "\n" + json.dumps({"tags": ["tag-b"]}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "processed" / "D3" / "yfcc-10M"
    summary = preprocess_d3_from_explicit_files(
        base_path=base,
        queries_path=queries,
        gt_ids_path=gt,
        filters_path=filters,
        payloads_path=payloads,
        out_dir=out_dir,
    )
    assert summary["task_type"] == "filtered_ann"
    assert len(_read_jsonl(out_dir / "filters.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "payloads.jsonl")) == 2


def test_preprocess_d3_explicit_rejects_non_2d_queries(tmp_path: Path) -> None:
    base = tmp_path / "base.npy"
    queries = tmp_path / "queries.npy"
    gt = tmp_path / "gt.npy"
    filters = tmp_path / "filters.jsonl"

    np.save(base, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(queries, np.asarray([1.0, 0.0], dtype=np.float32))
    np.save(gt, np.asarray([[0, 1]], dtype=np.int32))
    filters.write_text(json.dumps({"query_id": 0, "must_have_tags": ["tag-a"]}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "processed" / "D3" / "yfcc-10M"
    with pytest.raises(ValueError, match="queries array must be 2D"):
        preprocess_d3_from_explicit_files(
            base_path=base,
            queries_path=queries,
            gt_ids_path=gt,
            filters_path=filters,
            out_dir=out_dir,
        )


def test_preprocess_d3_yfcc_raw_to_explicit_writes_explicit_staging_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "dataset" / "D3" / "yfcc-10M"
    raw_dir.mkdir(parents=True)
    _write_u8bin(raw_dir / "base.10M.u8bin", np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.uint8))
    _write_u8bin(raw_dir / "query.public.100K.u8bin", np.asarray([[7, 8], [9, 10]], dtype=np.uint8))
    _write_ibin(raw_dir / "GT.public.ibin", np.asarray([[0, 1], [2, 1]], dtype=np.int32))
    _write_spmat(raw_dir / "base.metadata.10M.spmat", [[2, 4], [], [1]], ncol=8)
    _write_spmat(raw_dir / "query.metadata.public.100K.spmat", [[4], [1, 7]], ncol=8)

    out_dir = tmp_path / "explicit" / "D3" / "yfcc-10M"
    summary = preprocess_d3_yfcc_raw_to_explicit(
        dataset_dir=raw_dir,
        out_dir=out_dir,
        query_split="public",
        include_payloads=True,
    )

    assert summary["dataset_name"] == "yfcc-10M"
    assert summary["query_split"] == "public"
    np.testing.assert_allclose(
        np.load(out_dir / "base.npy", allow_pickle=False),
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(out_dir / "queries.npy", allow_pickle=False),
        np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(out_dir / "gt_ids.npy", allow_pickle=False),
        np.asarray([[0, 1], [2, 1]], dtype=np.int32),
    )
    assert _read_jsonl(out_dir / "filters.jsonl") == [
        {"query_id": 0, "must_have_tags": [4]},
        {"query_id": 1, "must_have_tags": [1, 7]},
    ]
    assert _read_jsonl(out_dir / "payloads.jsonl") == [
        {"tags": [2, 4]},
        {"tags": []},
        {"tags": [1]},
    ]


def test_preprocess_d3_yfcc_raw_then_d3_explicit_roundtrips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "dataset" / "D3" / "yfcc-10M"
    raw_dir.mkdir(parents=True)
    _write_u8bin(raw_dir / "base.10M.u8bin", np.asarray([[1, 2], [3, 4]], dtype=np.uint8))
    _write_u8bin(raw_dir / "query.public.100K.u8bin", np.asarray([[7, 8]], dtype=np.uint8))
    _write_ibin(raw_dir / "GT.public.ibin", np.asarray([[0, 1]], dtype=np.int32))
    _write_spmat(raw_dir / "base.metadata.10M.spmat", [[2], [5]], ncol=8)
    _write_spmat(raw_dir / "query.metadata.public.100K.spmat", [[2]], ncol=8)

    explicit_dir = tmp_path / "explicit" / "D3" / "yfcc-10M"
    preprocess_d3_yfcc_raw_to_explicit(
        dataset_dir=raw_dir,
        out_dir=explicit_dir,
        query_split="public",
        include_payloads=True,
    )

    processed_dir = tmp_path / "processed" / "D3" / "yfcc-10M"
    summary = preprocess_d3_from_explicit_files(
        base_path=explicit_dir / "base.npy",
        queries_path=explicit_dir / "queries.npy",
        gt_ids_path=explicit_dir / "gt_ids.npy",
        filters_path=explicit_dir / "filters.jsonl",
        payloads_path=explicit_dir / "payloads.jsonl",
        out_dir=processed_dir,
    )

    assert summary["task_type"] == "filtered_ann"
    assert (processed_dir / "meta.json").exists()
    assert _read_jsonl(processed_dir / "filters.jsonl") == [{"query_id": 0, "must_have_tags": [2]}]
    assert _read_jsonl(processed_dir / "payloads.jsonl") == [{"tags": [2]}, {"tags": [5]}]


def test_preprocess_d3_yfcc_raw_one_command_writes_canonical_meta(tmp_path: Path) -> None:
    raw_dir = tmp_path / "dataset" / "D3" / "yfcc-10M"
    raw_dir.mkdir(parents=True)
    _write_u8bin(raw_dir / "base.10M.u8bin", np.asarray([[1, 2], [3, 4]], dtype=np.uint8))
    _write_u8bin(raw_dir / "query.public.100K.u8bin", np.asarray([[7, 8]], dtype=np.uint8))
    _write_ibin(raw_dir / "GT.public.ibin", np.asarray([[0, 1]], dtype=np.int32))
    _write_spmat(raw_dir / "base.metadata.10M.spmat", [[2], [5]], ncol=8)
    _write_spmat(raw_dir / "query.metadata.public.100K.spmat", [[2]], ncol=8)

    processed_dir = tmp_path / "processed" / "D3" / "yfcc-10M"
    summary = preprocess_d3_yfcc_raw(
        dataset_dir=raw_dir,
        out_dir=processed_dir,
        query_split="public",
        include_payloads=True,
    )

    assert summary["task_type"] == "filtered_ann"
    meta = _read_json(processed_dir / "meta.json")
    assert meta["family"] == "D3"
    assert meta["task_type"] == "filtered_ann"
    assert meta["extra"]["query_split"] == "public"
    assert _read_jsonl(processed_dir / "filters.jsonl") == [{"query_id": 0, "must_have_tags": [2]}]


def test_preprocess_d3_explicit_ignores_empty_payloads_file(tmp_path: Path) -> None:
    base = tmp_path / "base.npy"
    queries = tmp_path / "queries.npy"
    gt = tmp_path / "gt.npy"
    filters = tmp_path / "filters.jsonl"
    payloads = tmp_path / "payloads.jsonl"

    np.save(base, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(queries, np.asarray([[1.0, 0.0]], dtype=np.float32))
    np.save(gt, np.asarray([[0, 1]], dtype=np.int32))
    filters.write_text(json.dumps({"query_id": 0, "must_have_tags": ["tag-a"]}) + "\n", encoding="utf-8")
    payloads.write_text("", encoding="utf-8")

    out_dir = tmp_path / "processed" / "D3" / "yfcc-10M"
    summary = preprocess_d3_from_explicit_files(
        base_path=base,
        queries_path=queries,
        gt_ids_path=gt,
        filters_path=filters,
        payloads_path=payloads,
        out_dir=out_dir,
    )
    assert summary["task_type"] == "filtered_ann"
    assert not (out_dir / "payloads.jsonl").exists()


def test_preprocess_crag_small_slice_rejects_overlap_ge_chunk_chars(tmp_path: Path) -> None:
    source = tmp_path / "crag.first_1.jsonl.bz2"
    with bz2.open(source, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "interaction_id": "1",
                    "query": "weather report",
                    "search_results": [
                        {
                            "page_name": "Forecast",
                            "page_result": "<html><body>weather report today</body></html>",
                            "page_url": "https://example.com/weather",
                        }
                    ],
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="overlap must be < chunk_chars"):
        preprocess_crag_small_slice(
            crag_path=source,
            out_dir=tmp_path / "processed" / "D4" / "crag" / "small_slice",
            max_examples=1,
            chunk_chars=50,
            overlap=50,
        )


def test_preprocess_beir_dataset_prefixes_ids(tmp_path: Path, monkeypatch) -> None:
    fake_loader_module = types.ModuleType("beir.datasets.data_loader")

    class _FakeLoader:
        def __init__(self, data_folder: str) -> None:
            self.data_folder = data_folder

        def load(self, split: str = "test"):
            assert split == "test"
            corpus = {"d1": {"title": "Bond", "text": "bond market"}}
            queries = {"q1": "bond market"}
            qrels = {"q1": {"d1": 2}}
            return corpus, queries, qrels

    fake_loader_module.GenericDataLoader = _FakeLoader
    fake_datasets_module = types.ModuleType("beir.datasets")
    fake_datasets_module.data_loader = fake_loader_module
    fake_beir_module = types.ModuleType("beir")
    fake_beir_module.datasets = fake_datasets_module

    monkeypatch.setitem(sys.modules, "beir", fake_beir_module)
    monkeypatch.setitem(sys.modules, "beir.datasets", fake_datasets_module)
    monkeypatch.setitem(sys.modules, "beir.datasets.data_loader", fake_loader_module)

    out_dir = tmp_path / "processed" / "D4" / "beir" / "fiqa"
    summary = preprocess_beir_dataset(
        dataset_dir=tmp_path / "raw" / "fiqa",
        out_dir=out_dir,
        dataset_name="fiqa",
        split="test",
    )
    assert summary["dataset_name"] == "fiqa"
    corpus = _read_jsonl(out_dir / "corpus.jsonl")
    queries = _read_jsonl(out_dir / "queries.jsonl")
    qrels = (out_dir / "qrels.tsv").read_text(encoding="utf-8")
    assert corpus[0]["doc_id"] == "fiqa::doc::d1"
    assert queries[0]["query_id"] == "fiqa::q::q1"
    assert "fiqa::q::q1\tfiqa::doc::d1\t2" in qrels


def test_preprocess_crag_small_slice_creates_weak_labels(tmp_path: Path) -> None:
    source = tmp_path / "crag.first_2.jsonl.bz2"
    rows = [
        {
            "interaction_id": "1",
            "query": "weather report",
            "search_results": [
                {
                    "page_name": "Forecast",
                    "page_result": "<html><body>weather report today</body></html>",
                    "page_url": "https://example.com/weather",
                }
            ],
        }
    ]
    with bz2.open(source, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "processed" / "D4" / "crag" / "small_slice"
    summary = preprocess_crag_small_slice(
        crag_path=source,
        out_dir=out_dir,
        max_examples=1,
        chunk_chars=50,
        overlap=10,
    )
    assert summary["task_type"] == "text_retrieval_weak"
    corpus = _read_jsonl(out_dir / "corpus.jsonl")
    queries = _read_jsonl(out_dir / "queries.jsonl")
    qrels = (out_dir / "qrels.tsv").read_text(encoding="utf-8")
    assert corpus[0]["doc_id"].startswith("crag_small_slice::doc::")
    assert queries[0]["query_id"] == "crag_small_slice::q::1"
    assert "crag_small_slice::q::1" in qrels
