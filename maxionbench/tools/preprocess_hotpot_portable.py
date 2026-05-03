"""Build the bounded HotpotQA-MaxionBench corpus from the official dev distractor set."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Any

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION
from maxionbench.tools.preprocess_datasets import ensure_dir, write_json, write_jsonl, write_qrels_tsv


def preprocess_hotpot_portable(
    *,
    input_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    rows = json.loads(input_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"HotpotQA input must be a JSON list: {input_path}")

    docs_by_key: dict[tuple[str, str], str] = {}
    corpus_rows: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    qrels_rows: list[tuple[str, str, int]] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        raw_qid = str(row.get("_id") or row.get("id") or idx).strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        context = row.get("context")
        supporting = row.get("supporting_facts")
        if not raw_qid or not question:
            continue
        if not isinstance(context, list) or not isinstance(supporting, list):
            continue

        query_id = f"hotpotqa_portable::q::{raw_qid}"
        query_rows.append(
            {
                "query_id": query_id,
                "text": question,
                "answer": answer,
                "source": "hotpotqa_portable",
            }
        )

        title_to_doc_id: dict[str, str] = {}
        for item in context:
            if not isinstance(item, list) or len(item) != 2:
                continue
            title = str(item[0] or "").strip()
            sentences = item[1]
            if not title or not isinstance(sentences, list):
                continue
            text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
            if not text:
                continue
            key = (title, text)
            doc_id = docs_by_key.get(key)
            if doc_id is None:
                digest = hashlib.sha256(f"{title}|{text}".encode("utf-8")).hexdigest()[:16]
                doc_id = f"hotpotqa_portable::doc::{digest}"
                docs_by_key[key] = doc_id
                corpus_rows[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_portable",
                }
            title_to_doc_id.setdefault(title, doc_id)

        seen_doc_ids: set[str] = set()
        for item in supporting:
            if not isinstance(item, list) or not item:
                continue
            title = str(item[0] or "").strip()
            doc_id = title_to_doc_id.get(title)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            qrels_rows.append((query_id, doc_id, 1))

    ensure_dir(out_dir)
    meta = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "preprocess_version": "0.1",
        "dataset_name": "hotpotqa-maxionbench",
        "family": "D4",
        "task_type": "text_retrieval_strict",
        "metric": "evidence_coverage",
        "num_docs": len(corpus_rows),
        "num_queries": len(query_rows),
        "source_path": str(input_path.expanduser().resolve()),
        "extra": {
            "source_family": "HotpotQA",
            "split": "dev_distractor",
            "retrieval_unit": "context_paragraph",
        },
    }
    write_json(out_dir / "meta.json", meta)
    write_jsonl(out_dir / "corpus.jsonl", corpus_rows.values())
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_qrels_tsv(out_dir / "qrels.tsv", qrels_rows)

    manifest = {
        "dataset_name": "hotpotqa-maxionbench",
        "question_count": len(query_rows),
        "doc_count": len(corpus_rows),
        "qrel_count": len(qrels_rows),
        "source_file": str(input_path.expanduser().resolve()),
    }
    write_json(out_dir / "manifest.json", manifest)

    checksums = {}
    for rel_path in ("meta.json", "corpus.jsonl", "queries.jsonl", "qrels.tsv", "manifest.json"):
        file_path = out_dir / rel_path
        checksums[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    write_json(out_dir / "checksums.json", checksums)
    return {
        "output_dir": str(out_dir.expanduser().resolve()),
        "dataset_name": "hotpotqa-maxionbench",
        "num_docs": len(corpus_rows),
        "num_queries": len(query_rows),
        "num_qrels": len(qrels_rows),
    }


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Build the bounded HotpotQA-MaxionBench corpus from the official dev distractor set")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = preprocess_hotpot_portable(
        input_path=Path(args.input).expanduser(),
        out_dir=Path(args.out).expanduser(),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
