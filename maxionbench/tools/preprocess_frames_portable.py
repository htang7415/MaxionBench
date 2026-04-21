"""Build the bounded FRAMES-portable corpus from local FRAMES and KILT-style inputs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION
from maxionbench.tools.preprocess_datasets import ensure_dir, write_json, write_jsonl, write_qrels_tsv


@dataclass(frozen=True)
class _DocRecord:
    doc_id: str
    page_id: str
    url: str
    text: str
    source: str


@dataclass(frozen=True)
class _QuestionRecord:
    query_id: str
    text: str
    gold_evidence: tuple[dict[str, str], ...]


def preprocess_frames_portable(
    *,
    frames_root: Path,
    kilt_root: Path,
    out_dir: Path,
    same_page_negatives: int = 6,
    cross_question_negatives: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    if same_page_negatives < 0:
        raise ValueError("same_page_negatives must be >= 0")
    if cross_question_negatives < 0:
        raise ValueError("cross_question_negatives must be >= 0")

    questions = _load_frames_questions(frames_root)
    page_chunks = _load_kilt_pages(kilt_root)

    docs_by_key: dict[tuple[str, str], _DocRecord] = {}
    corpus_rows: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    qrels_rows: list[tuple[str, str, int]] = []
    gold_doc_ids_by_query: dict[str, list[str]] = {}
    all_gold_doc_ids: list[str] = []

    for question in questions:
        query_rows.append({"query_id": question.query_id, "text": question.text})
        gold_doc_ids: list[str] = []
        for evidence in question.gold_evidence:
            doc = _register_doc(
                docs_by_key=docs_by_key,
                corpus_rows=corpus_rows,
                page_id=evidence["page_id"],
                url=evidence["url"],
                text=evidence["text"],
                source="gold",
            )
            if doc.doc_id not in gold_doc_ids:
                gold_doc_ids.append(doc.doc_id)
                qrels_rows.append((question.query_id, doc.doc_id, 1))
            if doc.doc_id not in all_gold_doc_ids:
                all_gold_doc_ids.append(doc.doc_id)
        gold_doc_ids_by_query[question.query_id] = gold_doc_ids

    for question in questions:
        gold_doc_ids = set(gold_doc_ids_by_query[question.query_id])
        for evidence in question.gold_evidence:
            page_id = evidence["page_id"]
            page_candidates = _page_negative_candidates(
                chunks=page_chunks.get(page_id, ()),
                gold_text=evidence["text"],
            )
            sampled_page_negatives = _sample_rows(
                rows=page_candidates,
                count=same_page_negatives,
                seed_payload=f"{seed}:same-page:{question.query_id}:{page_id}:{_text_hash(evidence['text'])}",
            )
            for row in sampled_page_negatives:
                doc = _register_doc(
                    docs_by_key=docs_by_key,
                    corpus_rows=corpus_rows,
                    page_id=page_id,
                    url=str(row.get("url") or f"kilt://{page_id}"),
                    text=str(row["text"]),
                    source="same_page_negative",
                )
                gold_doc_ids.add(doc.doc_id)

        cross_question_pool = [
            doc_id
            for doc_id in all_gold_doc_ids
            if doc_id not in gold_doc_ids_by_query[question.query_id]
        ]
        sampled_cross = _sample_rows(
            rows=[{"doc_id": doc_id} for doc_id in cross_question_pool],
            count=cross_question_negatives,
            seed_payload=f"{seed}:cross-question:{question.query_id}",
        )
        for row in sampled_cross:
            doc_id = str(row["doc_id"])
            if doc_id not in corpus_rows:
                raise ValueError(f"cross-question doc id missing from corpus registry: {doc_id}")

    ensure_dir(out_dir)
    meta = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "preprocess_version": "0.1",
        "dataset_name": "frames-portable",
        "family": "D4",
        "task_type": "text_retrieval_strict",
        "metric": "evidence_coverage",
        "num_docs": len(corpus_rows),
        "num_queries": len(query_rows),
        "same_page_negatives": same_page_negatives,
        "cross_question_negatives": cross_question_negatives,
        "seed": seed,
        "source_path": json.dumps(
            {
                "frames_root": str(frames_root.expanduser().resolve()),
                "kilt_root": str(kilt_root.expanduser().resolve()),
            },
            sort_keys=True,
        ),
        "extra": {
            "bounded_runtime_dependency": True,
            "dedup_policy": "normalized_url + text_hash",
        },
    }
    write_json(out_dir / "meta.json", meta)
    write_jsonl(out_dir / "corpus.jsonl", corpus_rows.values())
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_qrels_tsv(out_dir / "qrels.tsv", qrels_rows)

    manifest = {
        "dataset_name": "frames-portable",
        "question_count": len(query_rows),
        "doc_count": len(corpus_rows),
        "gold_doc_count": len(all_gold_doc_ids),
        "same_page_negatives": same_page_negatives,
        "cross_question_negatives": cross_question_negatives,
        "seed": seed,
        "source_files": {
            "frames": str(_discover_frames_file(frames_root)),
            "kilt": str(_discover_kilt_file(kilt_root)),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    checksums = {}
    for rel_path in ("meta.json", "corpus.jsonl", "queries.jsonl", "qrels.tsv", "manifest.json"):
        checksums[rel_path] = _file_sha256(out_dir / rel_path)
    write_json(out_dir / "checksums.json", checksums)
    return {
        "output_dir": str(out_dir.expanduser().resolve()),
        "dataset_name": "frames-portable",
        "num_docs": len(corpus_rows),
        "num_queries": len(query_rows),
        "same_page_negatives": same_page_negatives,
        "cross_question_negatives": cross_question_negatives,
    }


def _load_frames_questions(frames_root: Path) -> list[_QuestionRecord]:
    frames_file = _discover_frames_file(frames_root)
    payload = _load_jsonish(frames_file)
    rows = payload.get("questions") if isinstance(payload, dict) and "questions" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"FRAMES input must be a list of question rows: {frames_file}")
    questions: list[_QuestionRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = _first_nonempty(row, ("query_id", "question_id", "id", "qid"))
        text = _first_nonempty(row, ("text", "query", "question"))
        evidence_rows = row.get("gold_evidence") or row.get("evidence") or row.get("supporting_passages") or row.get("gold_passages")
        if not query_id or not text or not isinstance(evidence_rows, list):
            continue
        evidence_payload: list[dict[str, str]] = []
        for item in evidence_rows:
            if not isinstance(item, dict):
                continue
            page_id = _first_nonempty(item, ("page_id", "wikipedia_id", "source_page_id", "page", "page_title"))
            evidence_text = _first_nonempty(item, ("text", "passage", "content"))
            if not page_id or not evidence_text:
                continue
            evidence_payload.append(
                {
                    "page_id": page_id,
                    "url": _first_nonempty(item, ("url", "page_url")) or f"kilt://{page_id}",
                    "text": evidence_text,
                }
            )
        if evidence_payload:
            questions.append(
                _QuestionRecord(
                    query_id=query_id,
                    text=text,
                    gold_evidence=tuple(evidence_payload),
                )
            )
    if not questions:
        raise ValueError(f"No valid FRAMES questions found under {frames_root}")
    return questions


def _load_kilt_pages(kilt_root: Path) -> dict[str, tuple[dict[str, str], ...]]:
    kilt_file = _discover_kilt_file(kilt_root)
    rows = _load_jsonl_or_array(kilt_file)
    pages: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        page_id = _first_nonempty(row, ("page_id", "wikipedia_id", "id", "page"))
        if not page_id:
            continue
        chunks = row.get("chunks")
        if isinstance(chunks, list):
            for item in chunks:
                normalized = _normalize_chunk(item, default_page_id=page_id)
                if normalized is not None:
                    pages.setdefault(page_id, []).append(normalized)
            continue
        normalized_row = _normalize_chunk(row, default_page_id=page_id)
        if normalized_row is not None:
            pages.setdefault(page_id, []).append(normalized_row)
    return {key: tuple(value) for key, value in pages.items()}


def _page_negative_candidates(*, chunks: Iterable[dict[str, str]], gold_text: str) -> list[dict[str, str]]:
    gold_hash = _text_hash(gold_text)
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text or _text_hash(text) == gold_hash:
            continue
        key = (_normalize_url(str(chunk.get("url") or "")), _text_hash(text))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"text": text, "url": str(chunk.get("url") or "")})
    candidates.sort(key=lambda row: (_normalize_url(row["url"]), _text_hash(row["text"])))
    return candidates


def _sample_rows(*, rows: list[dict[str, Any]], count: int, seed_payload: str) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    rng = random.Random(_stable_int(seed_payload))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    picked = sorted(indices[:count])
    return [rows[idx] for idx in picked]


def _register_doc(
    *,
    docs_by_key: dict[tuple[str, str], _DocRecord],
    corpus_rows: dict[str, dict[str, Any]],
    page_id: str,
    url: str,
    text: str,
    source: str,
) -> _DocRecord:
    normalized_url = _normalize_url(url)
    digest = _text_hash(text)
    key = (normalized_url, digest)
    existing = docs_by_key.get(key)
    if existing is not None:
        return existing
    doc_id = f"frames_portable::doc::{hashlib.sha256(f'{normalized_url}|{digest}'.encode('utf-8')).hexdigest()[:16]}"
    record = _DocRecord(
        doc_id=doc_id,
        page_id=page_id,
        url=normalized_url or f"kilt://{page_id}",
        text=text.strip(),
        source=source,
    )
    docs_by_key[key] = record
    corpus_rows[doc_id] = {
        "doc_id": doc_id,
        "title": page_id,
        "text": text.strip(),
        "page_id": page_id,
        "url": record.url,
        "source": source,
    }
    return record


def _discover_frames_file(frames_root: Path) -> Path:
    root = frames_root.expanduser().resolve()
    candidates = [
        root / "questions.jsonl",
        root / "frames.jsonl",
        root / "questions.json",
        root / "frames.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    jsonish = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    if len(jsonish) == 1:
        return jsonish[0]
    raise FileNotFoundError(f"Could not locate FRAMES question file under {root}")


def _discover_kilt_file(kilt_root: Path) -> Path:
    root = kilt_root.expanduser().resolve()
    candidates = [
        root / "pages.jsonl",
        root / "passages.jsonl",
        root / "pages.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    jsonish = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    if len(jsonish) == 1:
        return jsonish[0]
    raise FileNotFoundError(f"Could not locate KILT page file under {root}")


def _load_jsonish(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _load_jsonl_or_array(path: Path) -> list[Any]:
    payload = _load_jsonish(path)
    if isinstance(payload, dict) and "pages" in payload:
        pages = payload["pages"]
        if isinstance(pages, list):
            return pages
    if isinstance(payload, list):
        return payload
    raise ValueError(f"KILT input must be a list of page rows: {path}")


def _normalize_chunk(item: Any, *, default_page_id: str) -> dict[str, str] | None:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"page_id": default_page_id, "text": text, "url": f"kilt://{default_page_id}"}
    if not isinstance(item, dict):
        return None
    text = _first_nonempty(item, ("text", "passage", "content"))
    if not text:
        return None
    return {
        "page_id": _first_nonempty(item, ("page_id", "wikipedia_id", "id", "page")) or default_page_id,
        "text": text,
        "url": _first_nonempty(item, ("url", "page_url")) or f"kilt://{default_page_id}",
    }


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_url(url: str) -> str:
    return url.strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _stable_int(payload: str) -> int:
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Build the bounded FRAMES-portable corpus from local FRAMES and KILT-style inputs")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--kilt-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--same-page-negatives", type=int, default=6)
    parser.add_argument("--cross-question-negatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = preprocess_frames_portable(
        frames_root=Path(args.frames_root).expanduser(),
        kilt_root=Path(args.kilt_root).expanduser(),
        out_dir=Path(args.out).expanduser(),
        same_page_negatives=int(args.same_page_negatives),
        cross_question_negatives=int(args.cross_question_negatives),
        seed=int(args.seed),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
