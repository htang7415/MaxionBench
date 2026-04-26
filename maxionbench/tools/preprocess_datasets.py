"""Canonical preprocessing for portable text datasets."""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

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
    extra: dict[str, Any] | None = None


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
        {"doc_id": doc_id, "title": "", "text": doc_text, "source": f"beir_{dataset_name}"}
        for doc_id, doc_text in zip(bundle.doc_ids, bundle.doc_texts)
    ]
    query_rows = [
        {"query_id": query_id, "text": query_text, "source": f"beir_{dataset_name}"}
        for query_id, query_text in zip(bundle.query_ids, bundle.query_texts)
    ]
    qrel_rows = [
        (query_id, doc_id, int(relevance))
        for query_id, rels in bundle.qrels.items()
        for doc_id, relevance in rels.items()
    ]

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

        evidence_doc_id: str | None = None
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
                # Mark only the first chunk of the first page as the event evidence doc.
                # All other chunks are background distractors (no qrel entry).
                # This gives exactly one evidence doc per event, which is what S2
                # freshness probes insert and then try to retrieve.
                if evidence_doc_id is None:
                    evidence_doc_id = doc_id
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
    parser = ArgumentParser(description="Preprocess portable text datasets into the canonical MaxionBench layout")
    sub = parser.add_subparsers(dest="command", required=True)

    beir_parser = sub.add_parser("beir", help="Preprocess a BEIR dataset into corpus/query/qrels files")
    beir_parser.add_argument("--input", required=True)
    beir_parser.add_argument("--out", required=True)
    beir_parser.add_argument("--name", required=True)
    beir_parser.add_argument("--split", default="test")
    beir_parser.add_argument("--json", action="store_true")

    crag_parser = sub.add_parser("crag", help="Preprocess a CRAG slice into weak-label files")
    crag_parser.add_argument("--input", required=True)
    crag_parser.add_argument("--out", required=True)
    crag_parser.add_argument("--max-examples", type=int, default=500)
    crag_parser.add_argument("--chunk-chars", type=int, default=1200)
    crag_parser.add_argument("--overlap", type=int, default=150)
    crag_parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "beir":
        summary = preprocess_beir_dataset(
            dataset_dir=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            dataset_name=str(args.name),
            split=str(args.split),
        )
    else:
        summary = preprocess_crag_small_slice(
            crag_path=Path(args.input).expanduser(),
            out_dir=Path(args.out).expanduser(),
            max_examples=int(args.max_examples),
            chunk_chars=int(args.chunk_chars),
            overlap=int(args.overlap),
        )

    if args.json:
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
    return re.sub(r"\s+", " ", text).strip()


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


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()
