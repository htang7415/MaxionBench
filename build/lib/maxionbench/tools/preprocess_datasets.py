"""Canonical dataset preprocessing for raw -> processed MaxionBench artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.lib.format import open_memmap

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION

_PREPROCESS_VERSION = "0.1"
_TAG_RE = re.compile(r"<[^>]+>")
_YFCC_PRIVATE_QUERY_RE = re.compile(r"^query\.private\.(?P<token>[0-9]+)\.100K\.u8bin$")
_YFCC_PRIVATE_GT_RE = re.compile(r"^GT\.private\.(?P<token>[0-9]+)\.ibin$")
_YFCC_PRIVATE_META_RE = re.compile(r"^query\.metadata\.private\.(?P<token>[0-9]+)\.100K\.spmat$")


@dataclass(frozen=True)
class DatasetMeta:
    schema_version: str
    preprocess_version: str
    dataset_name: str
    family: str
    task_type: str
    metric: str
    num_base: int
    num_queries: int
    dim: int
    source_path: str
    created_at_utc: str
    ground_truth_k: int | None = None
    dtype: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class _YfccQueryArtifacts:
    split: str
    queries_path: Path
    gt_ids_path: Path
    metadata_path: Path
    token: str | None = None


def preprocess_ann_hdf5(
    *,
    input_path: Path,
    out_dir: Path,
    family: str,
    dataset_name: str,
    metric: str,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    with h5py.File(input_path, "r") as handle:
        base = np.asarray(_read_first(handle, ["train", "vectors", "embeddings"]), dtype=np.float32)
        queries = np.asarray(_read_first(handle, ["test", "queries"]), dtype=np.float32)
        gt_ids = np.asarray(_read_first(handle, ["neighbors"]), dtype=np.int32)
        gt_dists = np.asarray(handle["distances"], dtype=np.float32) if "distances" in handle else None

    np.save(out_dir / "base.npy", base)
    np.save(out_dir / "queries.npy", queries)
    np.save(out_dir / "gt_ids.npy", gt_ids)
    if gt_dists is not None:
        np.save(out_dir / "gt_dists.npy", gt_dists)

    meta = DatasetMeta(
        schema_version=PROCESSED_SCHEMA_VERSION,
        preprocess_version=_PREPROCESS_VERSION,
        dataset_name=dataset_name,
        family=family,
        task_type="ann",
        metric=metric,
        num_base=int(base.shape[0]),
        num_queries=int(queries.shape[0]),
        dim=int(base.shape[1]),
        ground_truth_k=int(gt_ids.shape[1]),
        dtype=str(base.dtype),
        source_path=str(input_path.expanduser().resolve()),
        created_at_utc=_utc_now_iso(),
    )
    write_json(out_dir / "meta.json", asdict(meta))
    return {
        "output_dir": str(out_dir.resolve()),
        "family": family,
        "dataset_name": dataset_name,
        "task_type": meta.task_type,
        "num_base": meta.num_base,
        "num_queries": meta.num_queries,
        "dim": meta.dim,
    }


def preprocess_d3_from_explicit_files(
    *,
    base_path: Path,
    queries_path: Path,
    gt_ids_path: Path,
    filters_path: Path,
    out_dir: Path,
    payloads_path: Path | None = None,
    metric: str = "l2",
) -> dict[str, Any]:
    ensure_dir(out_dir)
    base = np.asarray(_load_array_file(base_path, mmap=True))
    queries = np.asarray(_load_array_file(queries_path, mmap=True))
    gt_ids = np.asarray(_load_array_file(gt_ids_path, mmap=True))
    if base.ndim != 2:
        raise ValueError(f"D3 base array must be 2D [N, D]; got shape={tuple(base.shape)}")
    if queries.ndim != 2:
        raise ValueError(f"D3 queries array must be 2D [Q, D]; got shape={tuple(queries.shape)}")
    if gt_ids.ndim != 2:
        raise ValueError(f"D3 gt_ids array must be 2D [Q, K]; got shape={tuple(gt_ids.shape)}")
    if base.shape[1] != queries.shape[1]:
        raise ValueError(
            f"D3 base/query dimension mismatch: base dim={int(base.shape[1])}, queries dim={int(queries.shape[1])}"
        )
    if int(gt_ids.shape[0]) < int(queries.shape[0]):
        raise ValueError(
            f"D3 gt_ids rows must cover all queries: gt rows={int(gt_ids.shape[0])}, queries={int(queries.shape[0])}"
        )
    _write_array_file(base_path, out_dir / "base.npy", dtype=np.float32)
    _write_array_file(queries_path, out_dir / "queries.npy", dtype=np.float32)
    _write_array_file(gt_ids_path, out_dir / "gt_ids.npy", dtype=np.int32)

    written_filters = _copy_jsonl_rows(
        filters_path,
        out_dir / "filters.jsonl",
        max_rows=int(queries.shape[0]),
    )
    if written_filters < int(queries.shape[0]):
        raise ValueError("filters file must include at least one row per query")

    payload_out_path = out_dir / "payloads.jsonl"
    payload_rows = 0
    if payloads_path is not None:
        payload_rows = _copy_jsonl_rows(payloads_path, payload_out_path)
    if payload_rows == 0 and payload_out_path.exists():
        payload_out_path.unlink()

    meta = DatasetMeta(
        schema_version=PROCESSED_SCHEMA_VERSION,
        preprocess_version=_PREPROCESS_VERSION,
        dataset_name="yfcc-10M",
        family="D3",
        task_type="filtered_ann",
        metric=metric,
        num_base=int(base.shape[0]),
        num_queries=int(queries.shape[0]),
        dim=int(base.shape[1]),
        ground_truth_k=int(gt_ids.shape[1]),
        dtype=str(base.dtype),
        source_path=json.dumps(
            {
                "base": str(base_path.expanduser().resolve()),
                "queries": str(queries_path.expanduser().resolve()),
                "gt_ids": str(gt_ids_path.expanduser().resolve()),
                "filters": str(filters_path.expanduser().resolve()),
                "payloads": str(payloads_path.expanduser().resolve()) if payloads_path is not None else None,
            },
            sort_keys=True,
        ),
        created_at_utc=_utc_now_iso(),
        extra={"filter_type": "tags"},
    )
    write_json(out_dir / "meta.json", asdict(meta))
    return {
        "output_dir": str(out_dir.resolve()),
        "family": "D3",
        "dataset_name": "yfcc-10M",
        "task_type": meta.task_type,
        "num_base": meta.num_base,
        "num_queries": meta.num_queries,
        "dim": meta.dim,
    }


def preprocess_d3_yfcc_raw_to_explicit(
    *,
    dataset_dir: Path,
    out_dir: Path,
    query_split: str = "public",
    private_query_token: str | None = None,
    include_payloads: bool = True,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    root = dataset_dir.expanduser().resolve()
    base_path = root / "base.10M.u8bin"
    base_metadata_path = root / "base.metadata.10M.spmat"
    if not base_path.exists():
        raise FileNotFoundError(f"missing D3 base vectors: {base_path}")
    if not base_metadata_path.exists():
        raise FileNotFoundError(f"missing D3 base metadata: {base_metadata_path}")

    query_artifacts = _resolve_yfcc_query_artifacts(
        root,
        query_split=query_split,
        private_query_token=private_query_token,
    )
    base = _load_xbin_mmap(base_path, dtype=np.uint8)
    queries = _load_xbin_mmap(query_artifacts.queries_path, dtype=np.uint8)
    gt_ids = _read_ibin_array(query_artifacts.gt_ids_path)

    if base.ndim != 2:
        raise ValueError(f"raw D3 base vectors must be 2D [N, D]; got shape={tuple(base.shape)}")
    if queries.ndim != 2:
        raise ValueError(f"raw D3 query vectors must be 2D [Q, D]; got shape={tuple(queries.shape)}")
    if gt_ids.ndim != 2:
        raise ValueError(f"raw D3 ground-truth ids must be 2D [Q, K]; got shape={tuple(gt_ids.shape)}")
    if int(base.shape[1]) != int(queries.shape[1]):
        raise ValueError(
            f"raw D3 base/query dimension mismatch: base dim={int(base.shape[1])}, queries dim={int(queries.shape[1])}"
        )
    if int(gt_ids.shape[0]) != int(queries.shape[0]):
        raise ValueError(
            f"raw D3 gt rows must match query rows: gt rows={int(gt_ids.shape[0])}, queries={int(queries.shape[0])}"
        )

    base_meta_rows, base_meta_cols, base_meta_nnz = _read_spmat_header(base_metadata_path)
    query_meta_rows, query_meta_cols, query_meta_nnz = _read_spmat_header(query_artifacts.metadata_path)
    if base_meta_rows != int(base.shape[0]):
        raise ValueError(
            f"raw D3 base metadata rows must match base vectors: metadata rows={base_meta_rows}, base rows={int(base.shape[0])}"
        )
    if query_meta_rows != int(queries.shape[0]):
        raise ValueError(
            f"raw D3 query metadata rows must match query vectors: metadata rows={query_meta_rows}, queries={int(queries.shape[0])}"
        )
    if base_meta_cols != query_meta_cols:
        raise ValueError(
            f"raw D3 base/query metadata dimension mismatch: base cols={base_meta_cols}, query cols={query_meta_cols}"
        )

    _write_array_to_npy(base, out_dir / "base.npy", dtype=np.float32)
    _write_array_to_npy(queries, out_dir / "queries.npy", dtype=np.float32)
    _write_array_to_npy(gt_ids, out_dir / "gt_ids.npy", dtype=np.int32)
    _write_query_filters_from_spmat(
        query_artifacts.metadata_path,
        out_dir / "filters.jsonl",
        expected_rows=int(queries.shape[0]),
    )

    payload_path = out_dir / "payloads.jsonl"
    if include_payloads:
        _write_payloads_from_spmat(
            base_metadata_path,
            payload_path,
            expected_rows=int(base.shape[0]),
        )
    elif payload_path.exists():
        payload_path.unlink()

    write_json(
        out_dir / "source_manifest.json",
        {
            "dataset_name": "yfcc-10M",
            "source_layout": "big-ann-benchmarks-yfcc100M",
            "query_split": query_artifacts.split,
            "private_query_token": query_artifacts.token,
            "base_path": str(base_path),
            "queries_path": str(query_artifacts.queries_path),
            "gt_ids_path": str(query_artifacts.gt_ids_path),
            "base_metadata_path": str(base_metadata_path),
            "query_metadata_path": str(query_artifacts.metadata_path),
            "metadata_num_tags": int(base_meta_cols),
            "base_metadata_nnz": int(base_meta_nnz),
            "query_metadata_nnz": int(query_meta_nnz),
            "include_payloads": bool(include_payloads),
        },
    )
    return {
        "output_dir": str(out_dir.resolve()),
        "family": "D3",
        "dataset_name": "yfcc-10M",
        "query_split": query_artifacts.split,
        "num_base": int(base.shape[0]),
        "num_queries": int(queries.shape[0]),
        "dim": int(base.shape[1]),
        "ground_truth_k": int(gt_ids.shape[1]),
        "payloads_written": bool(include_payloads),
    }


def preprocess_d3_yfcc_raw(
    *,
    dataset_dir: Path,
    out_dir: Path,
    query_split: str = "public",
    private_query_token: str | None = None,
    include_payloads: bool = True,
    metric: str = "l2",
) -> dict[str, Any]:
    summary = preprocess_d3_yfcc_raw_to_explicit(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        query_split=query_split,
        private_query_token=private_query_token,
        include_payloads=include_payloads,
    )
    base = np.load(out_dir / "base.npy", mmap_mode="r", allow_pickle=False)
    queries = np.load(out_dir / "queries.npy", mmap_mode="r", allow_pickle=False)
    gt_ids = np.load(out_dir / "gt_ids.npy", mmap_mode="r", allow_pickle=False)
    source_manifest = json.loads((out_dir / "source_manifest.json").read_text(encoding="utf-8"))
    meta = DatasetMeta(
        schema_version=PROCESSED_SCHEMA_VERSION,
        preprocess_version=_PREPROCESS_VERSION,
        dataset_name="yfcc-10M",
        family="D3",
        task_type="filtered_ann",
        metric=metric,
        num_base=int(base.shape[0]),
        num_queries=int(queries.shape[0]),
        dim=int(base.shape[1]),
        ground_truth_k=int(gt_ids.shape[1]),
        dtype=str(base.dtype),
        source_path=json.dumps(source_manifest, sort_keys=True),
        created_at_utc=_utc_now_iso(),
        extra={
            "filter_type": "tags",
            "query_split": str(source_manifest.get("query_split", query_split)),
            "private_query_token": source_manifest.get("private_query_token"),
        },
    )
    write_json(out_dir / "meta.json", asdict(meta))
    return {
        "output_dir": str(out_dir.resolve()),
        "family": "D3",
        "dataset_name": "yfcc-10M",
        "task_type": meta.task_type,
        "query_split": str(source_manifest.get("query_split", query_split)),
        "num_base": meta.num_base,
        "num_queries": meta.num_queries,
        "dim": meta.dim,
        "ground_truth_k": meta.ground_truth_k,
        "payloads_written": bool(include_payloads),
    }


def preprocess_beir_dataset(
    *,
    dataset_dir: Path,
    out_dir: Path,
    dataset_name: str,
    split: str = "test",
) -> dict[str, Any]:
    from maxionbench.datasets.loaders.d4_text import _load_beir_subset_bundle

    ensure_dir(out_dir)
    bundle = _load_beir_subset_bundle(
        subset_dir=dataset_dir.expanduser().resolve(),
        subset_name=dataset_name,
        split=split,
        max_docs=1_000_000_000,
        max_queries=1_000_000_000,
    )

    corpus_rows = [
        {
            "doc_id": doc_id,
            "title": "",
            "text": doc_text,
            "source": f"beir_{dataset_name}",
        }
        for doc_id, doc_text in zip(bundle.doc_ids, bundle.doc_texts)
    ]

    query_rows = [
        {
            "query_id": query_id,
            "text": query_text,
            "source": f"beir_{dataset_name}",
        }
        for query_id, query_text in zip(bundle.query_ids, bundle.query_texts)
    ]

    qrel_rows: list[tuple[str, str, int]] = []
    for query_id, rels in bundle.qrels.items():
        for doc_id, relevance in rels.items():
            qrel_rows.append((query_id, doc_id, int(relevance)))

    write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_qrels_tsv(out_dir / "qrels.tsv", qrel_rows)

    meta = DatasetMeta(
        schema_version=PROCESSED_SCHEMA_VERSION,
        preprocess_version=_PREPROCESS_VERSION,
        dataset_name=dataset_name,
        family="D4",
        task_type="text_retrieval_strict",
        metric="ndcg_mrr_recall",
        num_base=len(corpus_rows),
        num_queries=len(query_rows),
        dim=0,
        source_path=str(dataset_dir.expanduser().resolve()),
        created_at_utc=_utc_now_iso(),
        extra={"source_family": "BEIR", "split": split},
    )
    write_json(out_dir / "meta.json", asdict(meta))
    return {
        "output_dir": str(out_dir.resolve()),
        "family": "D4",
        "dataset_name": dataset_name,
        "task_type": meta.task_type,
        "num_base": meta.num_base,
        "num_queries": meta.num_queries,
    }


def preprocess_crag_small_slice(
    *,
    crag_path: Path,
    out_dir: Path,
    max_examples: int = 500,
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    examples = list(load_jsonl_or_bz2(crag_path))[:max_examples]

    corpus_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[tuple[str, str, int]] = []

    for index, ex in enumerate(examples):
        raw_qid = str(ex.get("interaction_id") or ex.get("query_id") or index)
        qid = f"crag_small_slice::q::{raw_qid}"
        query_rows.append(
            {
                "query_id": qid,
                "text": str(ex.get("query") or ex.get("question") or "").strip(),
                "source": "crag_small_slice",
                "domain": str(ex.get("domain") or "").strip(),
                "query_time": str(ex.get("query_time") or "").strip(),
            }
        )

        for page_idx, page in enumerate(ex.get("search_results", [])):
            raw_html = str(page.get("page_result") or "")
            page_text = simple_html_to_text(raw_html)
            if not page_text:
                page_text = str(page.get("page_snippet") or page.get("text") or "").strip()
            if not page_text:
                continue
            for chunk_idx, chunk in enumerate(chunk_text(page_text, chunk_chars=chunk_chars, overlap=overlap)):
                doc_id = f"crag_small_slice::doc::{raw_qid}_p{page_idx}_c{chunk_idx}"
                corpus_rows.append(
                    {
                        "doc_id": doc_id,
                        "title": str(page.get("page_name") or page.get("title") or "").strip(),
                        "text": chunk,
                        "url": str(page.get("page_url") or "").strip(),
                        "source": "crag_small_slice",
                        "page_last_modified": str(page.get("page_last_modified") or "").strip(),
                    }
                )
                qrel_rows.append((qid, doc_id, 1))

    write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_qrels_tsv(out_dir / "qrels.tsv", qrel_rows)

    meta = DatasetMeta(
        schema_version=PROCESSED_SCHEMA_VERSION,
        preprocess_version=_PREPROCESS_VERSION,
        dataset_name="crag_small_slice",
        family="D4",
        task_type="text_retrieval_weak",
        metric="weak_labels",
        num_base=len(corpus_rows),
        num_queries=len(query_rows),
        dim=0,
        source_path=str(crag_path.expanduser().resolve()),
        created_at_utc=_utc_now_iso(),
        extra={
            "source_family": "CRAG",
            "label_policy": "weak_positive_from_search_results",
            "max_examples": max_examples,
            "chunk_chars": chunk_chars,
            "overlap": overlap,
        },
    )
    write_json(out_dir / "meta.json", asdict(meta))
    return {
        "output_dir": str(out_dir.resolve()),
        "family": "D4",
        "dataset_name": "crag_small_slice",
        "task_type": meta.task_type,
        "num_base": meta.num_base,
        "num_queries": meta.num_queries,
    }


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Preprocess raw datasets into the canonical MaxionBench processed layout")
    sub = parser.add_subparsers(dest="command", required=True)

    ann_parser = sub.add_parser("ann-hdf5", help="Preprocess ann-benchmarks style HDF5 into canonical ANN files")
    ann_parser.add_argument("--input", required=True)
    ann_parser.add_argument("--out", required=True)
    ann_parser.add_argument("--family", required=True)
    ann_parser.add_argument("--name", required=True)
    ann_parser.add_argument("--metric", required=True)
    ann_parser.add_argument("--json", action="store_true")

    d3_raw_parser = sub.add_parser(
        "d3-yfcc-raw",
        help="Convert raw YFCC big-ann files into explicit D3 arrays/filter inputs",
    )
    d3_raw_parser.add_argument("--input", required=True)
    d3_raw_parser.add_argument("--out", required=True)
    d3_raw_parser.add_argument("--query-split", choices=["public", "private"], default="public")
    d3_raw_parser.add_argument("--private-query-token", default=None)
    d3_raw_parser.add_argument("--skip-payloads", action="store_true")
    d3_raw_parser.add_argument("--json", action="store_true")

    d3_yfcc_parser = sub.add_parser(
        "d3-yfcc",
        help="Preprocess raw YFCC big-ann files directly into canonical processed D3 files",
    )
    d3_yfcc_parser.add_argument("--input", required=True)
    d3_yfcc_parser.add_argument("--out", required=True)
    d3_yfcc_parser.add_argument("--query-split", choices=["public", "private"], default="public")
    d3_yfcc_parser.add_argument("--private-query-token", default=None)
    d3_yfcc_parser.add_argument("--skip-payloads", action="store_true")
    d3_yfcc_parser.add_argument("--metric", default="l2")
    d3_yfcc_parser.add_argument("--json", action="store_true")

    d3_parser = sub.add_parser("d3-explicit", help="Preprocess explicit D3 array/filter inputs into canonical files")
    d3_parser.add_argument("--base", required=True)
    d3_parser.add_argument("--queries", required=True)
    d3_parser.add_argument("--gt", required=True)
    d3_parser.add_argument("--filters", required=True)
    d3_parser.add_argument("--payloads", default=None)
    d3_parser.add_argument("--out", required=True)
    d3_parser.add_argument("--metric", default="l2")
    d3_parser.add_argument("--json", action="store_true")

    beir_parser = sub.add_parser("beir", help="Preprocess a BEIR dataset into canonical corpus/query/qrels files")
    beir_parser.add_argument("--input", required=True)
    beir_parser.add_argument("--out", required=True)
    beir_parser.add_argument("--name", required=True)
    beir_parser.add_argument("--split", default="test")
    beir_parser.add_argument("--json", action="store_true")

    crag_parser = sub.add_parser("crag", help="Preprocess a CRAG slice into canonical weak-label files")
    crag_parser.add_argument("--input", required=True)
    crag_parser.add_argument("--out", required=True)
    crag_parser.add_argument("--max-examples", type=int, default=500)
    crag_parser.add_argument("--chunk-chars", type=int, default=1200)
    crag_parser.add_argument("--overlap", type=int, default=150)
    crag_parser.add_argument("--json", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "ann-hdf5":
        summary = preprocess_ann_hdf5(
            input_path=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            family=str(args.family),
            dataset_name=str(args.name),
            metric=str(args.metric),
        )
        emit_json = bool(args.json)
    elif args.command == "d3-yfcc-raw":
        summary = preprocess_d3_yfcc_raw_to_explicit(
            dataset_dir=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            query_split=str(args.query_split),
            private_query_token=str(args.private_query_token) if args.private_query_token else None,
            include_payloads=not bool(args.skip_payloads),
        )
        emit_json = bool(args.json)
    elif args.command == "d3-yfcc":
        summary = preprocess_d3_yfcc_raw(
            dataset_dir=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            query_split=str(args.query_split),
            private_query_token=str(args.private_query_token) if args.private_query_token else None,
            include_payloads=not bool(args.skip_payloads),
            metric=str(args.metric),
        )
        emit_json = bool(args.json)
    elif args.command == "d3-explicit":
        summary = preprocess_d3_from_explicit_files(
            base_path=Path(args.base).expanduser(),
            queries_path=Path(args.queries).expanduser(),
            gt_ids_path=Path(args.gt).expanduser(),
            filters_path=Path(args.filters).expanduser(),
            payloads_path=Path(args.payloads).expanduser() if args.payloads else None,
            out_dir=Path(args.out).expanduser(),
            metric=str(args.metric),
        )
        emit_json = bool(args.json)
    elif args.command == "beir":
        summary = preprocess_beir_dataset(
            dataset_dir=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            dataset_name=str(args.name),
            split=str(args.split),
        )
        emit_json = bool(args.json)
    else:
        summary = preprocess_crag_small_slice(
            crag_path=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            max_examples=int(args.max_examples),
            chunk_chars=int(args.chunk_chars),
            overlap=int(args.overlap),
        )
        emit_json = bool(args.json)

    if emit_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary["output_dir"])
    return 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_qrels_tsv(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for row in rows:
            writer.writerow(row)


def load_jsonl_or_bz2(path: Path) -> Iterable[dict[str, Any]]:
    opener = bz2.open if path.suffix.lower() == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            row = line.strip()
            if not row:
                continue
            obj = json.loads(row)
            if isinstance(obj, dict):
                yield obj


def simple_html_to_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_chars:
        raise ValueError("overlap must be < chunk_chars")
    chunks: list[str] = []
    start = 0
    limit = len(text)
    while start < limit:
        end = min(start + chunk_chars, limit)
        chunks.append(text[start:end])
        if end == limit:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _read_first(handle: h5py.File, keys: Iterable[str]):
    for key in keys:
        if key in handle:
            return handle[key]
    available = sorted(handle.keys())
    raise KeyError(f"expected one of keys {list(keys)} but found {available}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            obj = json.loads(row)
            if isinstance(obj, dict):
                yield obj


def _load_array_file(path: Path, *, mmap: bool = False) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if isinstance(loaded, np.ndarray):
        return loaded
    if "vectors" in loaded:
        return np.asarray(loaded["vectors"])
    if len(loaded.files) == 1:
        return np.asarray(loaded[loaded.files[0]])
    raise ValueError(f"could not infer array from {path}; npz files must contain `vectors` or a single array")


def _write_array_file(src_path: Path, dst_path: Path, *, dtype: np.dtype[Any] | type[np.generic]) -> None:
    np_dtype = np.dtype(dtype)
    if src_path.suffix == ".npy":
        loaded = np.load(src_path, allow_pickle=False, mmap_mode="r")
        if isinstance(loaded, np.ndarray) and loaded.dtype == np_dtype:
            ensure_dir(dst_path.parent)
            shutil.copy2(src_path, dst_path)
            return
    _write_array_to_npy(_load_array_file(src_path, mmap=True), dst_path, dtype=np_dtype)


def _write_array_to_npy(
    array: np.ndarray,
    dst_path: Path,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    target_chunk_bytes: int = 64 * 1024 * 1024,
) -> None:
    src = np.asarray(array)
    np_dtype = np.dtype(dtype)
    ensure_dir(dst_path.parent)
    target = open_memmap(dst_path, mode="w+", dtype=np_dtype, shape=src.shape)
    if src.ndim == 0:
        target[...] = np.asarray(src, dtype=np_dtype)
        target.flush()
        del target
        return
    if src.ndim == 1:
        items_per_chunk = max(1, target_chunk_bytes // max(np_dtype.itemsize, 1))
        for start in range(0, int(src.shape[0]), items_per_chunk):
            end = min(int(src.shape[0]), start + items_per_chunk)
            target[start:end] = np.asarray(src[start:end], dtype=np_dtype)
        target.flush()
        del target
        return

    row_width = int(np.prod(src.shape[1:], dtype=np.int64))
    row_bytes = max(1, row_width * np_dtype.itemsize)
    rows_per_chunk = max(1, target_chunk_bytes // row_bytes)
    for start in range(0, int(src.shape[0]), rows_per_chunk):
        end = min(int(src.shape[0]), start + rows_per_chunk)
        target[start:end] = np.asarray(src[start:end], dtype=np_dtype)
    target.flush()
    del target


def _copy_jsonl_rows(path: Path, dst_path: Path, *, max_rows: int | None = None) -> int:
    ensure_dir(dst_path.parent)
    written = 0
    with dst_path.open("w", encoding="utf-8") as handle:
        for row in _read_jsonl(path):
            if max_rows is not None and written >= max_rows:
                break
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def _load_xbin_mmap(path: Path, *, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.shape[0] != 2:
        raise ValueError(f"invalid xbin header: {path}")
    n, dim = int(header[0]), int(header[1])
    expected_bytes = 8 + n * dim * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"xbin file size mismatch for {path}: expected {expected_bytes} bytes, found {actual_bytes}"
        )
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, dim))


def _read_ibin_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if header.shape[0] != 2:
            raise ValueError(f"invalid ibin header: {path}")
        n, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype=np.int32, count=n * dim)
    if data.size != n * dim:
        raise ValueError(f"ibin payload size mismatch for {path}: expected {n * dim} ints, found {data.size}")
    return data.reshape(n, dim)


def _read_spmat_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        sizes = np.fromfile(handle, dtype=np.int64, count=3)
    if sizes.shape[0] != 3:
        raise ValueError(f"invalid spmat header: {path}")
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def _read_spmat_index_fields(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    nrow, ncol, nnz = _read_spmat_header(path)
    ofs = 3 * np.dtype(np.int64).itemsize
    indptr = np.memmap(path, dtype=np.int64, mode="r", offset=ofs, shape=(nrow + 1,))
    ofs += indptr.nbytes
    indices = np.memmap(path, dtype=np.int32, mode="r", offset=ofs, shape=(nnz,))
    if int(indptr[-1]) != nnz:
        raise ValueError(f"invalid spmat indptr payload in {path}: expected nnz={nnz}, found tail={int(indptr[-1])}")
    if nnz and (int(indices.min()) < 0 or int(indices.max()) >= ncol):
        raise ValueError(f"invalid spmat indices in {path}: expected range 0..{ncol - 1}")
    return nrow, ncol, indices, indptr


def _write_query_filters_from_spmat(path: Path, out_path: Path, *, expected_rows: int) -> None:
    nrow, _, indices, indptr = _read_spmat_index_fields(path)
    if nrow != expected_rows:
        raise ValueError(f"query metadata rows must match query count: metadata rows={nrow}, queries={expected_rows}")
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        for row_idx in range(nrow):
            start = int(indptr[row_idx])
            end = int(indptr[row_idx + 1])
            row = {"query_id": row_idx, "must_have_tags": [int(tag) for tag in indices[start:end].tolist()]}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_payloads_from_spmat(path: Path, out_path: Path, *, expected_rows: int) -> None:
    nrow, _, indices, indptr = _read_spmat_index_fields(path)
    if nrow != expected_rows:
        raise ValueError(f"base metadata rows must match base vector count: metadata rows={nrow}, base rows={expected_rows}")
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        for row_idx in range(nrow):
            start = int(indptr[row_idx])
            end = int(indptr[row_idx + 1])
            row = {"tags": [int(tag) for tag in indices[start:end].tolist()]}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_yfcc_query_artifacts(
    dataset_dir: Path,
    *,
    query_split: str,
    private_query_token: str | None,
) -> _YfccQueryArtifacts:
    if query_split == "public":
        queries_path = dataset_dir / "query.public.100K.u8bin"
        gt_ids_path = dataset_dir / "GT.public.ibin"
        metadata_path = dataset_dir / "query.metadata.public.100K.spmat"
        for path in (queries_path, gt_ids_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)
        return _YfccQueryArtifacts(
            split="public",
            queries_path=queries_path,
            gt_ids_path=gt_ids_path,
            metadata_path=metadata_path,
        )

    query_candidates = _match_yfcc_tokened_files(dataset_dir, _YFCC_PRIVATE_QUERY_RE)
    gt_candidates = _match_yfcc_tokened_files(dataset_dir, _YFCC_PRIVATE_GT_RE)
    metadata_candidates = _match_yfcc_tokened_files(dataset_dir, _YFCC_PRIVATE_META_RE)
    common_tokens = sorted(set(query_candidates) & set(gt_candidates) & set(metadata_candidates))
    if not common_tokens:
        raise FileNotFoundError(f"no matching private YFCC query/GT/metadata bundle found under {dataset_dir}")
    token = str(private_query_token) if private_query_token else None
    if token is None:
        if len(common_tokens) != 1:
            raise ValueError(
                f"multiple private YFCC query tokens found under {dataset_dir}; pass --private-query-token "
                f"(available: {', '.join(common_tokens)})"
            )
        token = common_tokens[0]
    if token not in common_tokens:
        raise FileNotFoundError(
            f"private YFCC query token {token!r} not found under {dataset_dir}; available tokens: {', '.join(common_tokens)}"
        )
    return _YfccQueryArtifacts(
        split="private",
        queries_path=query_candidates[token],
        gt_ids_path=gt_candidates[token],
        metadata_path=metadata_candidates[token],
        token=token,
    )


def _match_yfcc_tokened_files(dataset_dir: Path, pattern: re.Pattern[str]) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    for path in sorted(dataset_dir.iterdir()):
        match = pattern.match(path.name)
        if match:
            matches[str(match.group("token"))] = path
    return matches


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
