from __future__ import annotations

import bz2
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np

from maxionbench.tools.preprocess_datasets import (
    preprocess_ann_hdf5,
    preprocess_beir_dataset,
    preprocess_crag_small_slice,
    preprocess_d3_from_explicit_files,
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
