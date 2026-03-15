"""Canonical dataset preprocessing for raw -> processed MaxionBench artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import h5py
import numpy as np

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION

_PREPROCESS_VERSION = "0.1"
_TAG_RE = re.compile(r"<[^>]+>")


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
    base = np.asarray(_load_array_file(base_path), dtype=np.float32)
    queries = np.asarray(_load_array_file(queries_path), dtype=np.float32)
    gt_ids = np.asarray(_load_array_file(gt_ids_path), dtype=np.int32)
    filters = list(_read_jsonl(filters_path))
    if len(filters) < int(queries.shape[0]):
        raise ValueError("filters file must include at least one row per query")
    payloads = list(_read_jsonl(payloads_path)) if payloads_path is not None else []

    np.save(out_dir / "base.npy", base)
    np.save(out_dir / "queries.npy", queries)
    np.save(out_dir / "gt_ids.npy", gt_ids)
    write_jsonl(out_dir / "filters.jsonl", filters[: int(queries.shape[0])])
    if payloads:
        write_jsonl(out_dir / "payloads.jsonl", payloads)

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


def preprocess_beir_dataset(
    *,
    dataset_dir: Path,
    out_dir: Path,
    dataset_name: str,
    split: str = "test",
) -> dict[str, Any]:
    from beir.datasets.data_loader import GenericDataLoader

    ensure_dir(out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)

    corpus_rows: list[dict[str, Any]] = []
    for doc_id, doc in corpus.items():
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or doc.get("contents") or "").strip()
        corpus_rows.append(
            {
                "doc_id": f"{dataset_name}::doc::{doc_id}",
                "title": title,
                "text": text,
                "source": f"beir_{dataset_name}",
            }
        )

    query_rows: list[dict[str, Any]] = []
    for query_id, text in queries.items():
        query_rows.append(
            {
                "query_id": f"{dataset_name}::q::{query_id}",
                "text": str(text),
                "source": f"beir_{dataset_name}",
            }
        )

    qrel_rows: list[tuple[str, str, int]] = []
    for query_id, rels in qrels.items():
        for doc_id, relevance in rels.items():
            qrel_rows.append((f"{dataset_name}::q::{query_id}", f"{dataset_name}::doc::{doc_id}", int(relevance)))

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


def _load_array_file(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    if "vectors" in loaded:
        return np.asarray(loaded["vectors"])
    if len(loaded.files) == 1:
        return np.asarray(loaded[loaded.files[0]])
    raise ValueError(f"could not infer array from {path}; npz files must contain `vectors` or a single array")


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
