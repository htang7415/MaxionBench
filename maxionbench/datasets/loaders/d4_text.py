"""Local-bundle loaders for D4 BEIR subsets and CRAG slice."""

from __future__ import annotations

import bz2
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from maxionbench.datasets.cache_integrity import verify_file_sha256

from .d4_synthetic import D4RetrievalDataset, compute_idf, tokenize_text


DEFAULT_BEIR_SUBSETS = ["scifact", "fiqa"]


@dataclass(frozen=True)
class _TextBundle:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]


def load_d4_from_local_bundles(
    *,
    vector_dim: int,
    seed: int,
    beir_root: Path | None = None,
    beir_subsets: Sequence[str] | None = None,
    beir_split: str = "test",
    crag_path: Path | None = None,
    crag_expected_sha256: str | None = None,
    include_crag: bool = True,
    max_docs: int = 200_000,
    max_queries: int = 5_000,
) -> D4RetrievalDataset:
    if vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    if max_docs < 1 or max_queries < 1:
        raise ValueError("max_docs and max_queries must be >= 1")

    bundles: list[_TextBundle] = []
    if beir_root is not None:
        subsets = list(beir_subsets or DEFAULT_BEIR_SUBSETS)
        for subset in subsets:
            subset_dir = (beir_root / subset).resolve()
            if not subset_dir.exists():
                continue
            bundles.append(
                _load_beir_subset_bundle(
                    subset_dir=subset_dir,
                    subset_name=subset,
                    split=beir_split,
                    max_docs=max_docs,
                    max_queries=max_queries,
                )
            )
    if include_crag and crag_path is not None and crag_path.exists():
        if crag_expected_sha256 is not None:
            verify_file_sha256(path=crag_path, expected_sha256=crag_expected_sha256, label="D4 d4_crag_path")
        bundles.append(
            _load_crag_bundle(
                crag_path=crag_path.resolve(),
                max_docs=max_docs,
                max_queries=max_queries,
            )
        )
    if not bundles:
        raise FileNotFoundError("No D4 local bundles were loaded. Check BEIR root and CRAG path.")
    merged = _merge_bundles(bundles=bundles, max_docs=max_docs, max_queries=max_queries)
    return _build_retrieval_dataset(bundle=merged, vector_dim=vector_dim, seed=seed)


def _load_beir_subset_bundle(
    *,
    subset_dir: Path,
    subset_name: str,
    split: str,
    max_docs: int,
    max_queries: int,
) -> _TextBundle:
    corpus_path = subset_dir / "corpus.jsonl"
    queries_path = subset_dir / "queries.jsonl"
    qrels_path = subset_dir / "qrels" / f"{split}.tsv"
    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            f"Missing BEIR files under {subset_dir}: corpus.jsonl / queries.jsonl / qrels/{split}.tsv"
        )

    raw_docs: dict[str, str] = {}
    for obj in _read_jsonl(corpus_path):
        raw_id = str(obj.get("_id") or obj.get("id") or obj.get("doc_id") or "").strip()
        if not raw_id:
            continue
        title = str(obj.get("title") or "").strip()
        text = str(obj.get("text") or obj.get("contents") or obj.get("content") or "").strip()
        full = f"{title} {text}".strip()
        if not full:
            continue
        pref_id = f"{subset_name}::doc::{raw_id}"
        if pref_id not in raw_docs:
            raw_docs[pref_id] = full
        if len(raw_docs) >= max_docs:
            break

    raw_queries: dict[str, str] = {}
    for obj in _read_jsonl(queries_path):
        raw_id = str(obj.get("_id") or obj.get("id") or obj.get("query_id") or "").strip()
        if not raw_id:
            continue
        text = str(obj.get("text") or obj.get("query") or obj.get("question") or "").strip()
        if not text:
            continue
        pref_qid = f"{subset_name}::q::{raw_id}"
        if pref_qid not in raw_queries:
            raw_queries[pref_qid] = text
        if len(raw_queries) >= max_queries:
            break

    qrels = _read_beir_qrels(
        qrels_path=qrels_path,
        subset_name=subset_name,
        allowed_qids=set(raw_queries.keys()),
        allowed_doc_ids=set(raw_docs.keys()),
    )
    # Keep only queries with at least one relevant doc.
    query_ids = [qid for qid in raw_queries.keys() if qid in qrels]
    query_texts = [raw_queries[qid] for qid in query_ids]
    doc_ids = list(raw_docs.keys())
    doc_texts = [raw_docs[doc_id] for doc_id in doc_ids]
    return _TextBundle(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )


def _read_beir_qrels(
    *,
    qrels_path: Path,
    subset_name: str,
    allowed_qids: set[str],
    allowed_doc_ids: set[str],
) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            row = line.strip()
            if not row:
                continue
            parts = row.split("\t")
            if line_no == 0 and any("query" in p.lower() for p in parts):
                continue
            qid_raw = ""
            did_raw = ""
            score_str = "0"
            if len(parts) >= 4:
                qid_raw = parts[0]
                did_raw = parts[2]
                score_str = parts[3]
            elif len(parts) >= 3:
                qid_raw = parts[0]
                did_raw = parts[1]
                score_str = parts[2]
            else:
                continue
            qid = f"{subset_name}::q::{qid_raw.strip()}"
            did = f"{subset_name}::doc::{did_raw.strip()}"
            if qid not in allowed_qids or did not in allowed_doc_ids:
                continue
            rel = _safe_rel(score_str)
            if rel <= 0:
                continue
            qrels.setdefault(qid, {})[did] = max(rel, qrels.get(qid, {}).get(did, 0))
    return qrels


def _load_crag_bundle(*, crag_path: Path, max_docs: int, max_queries: int) -> _TextBundle:
    docs: dict[str, str] = {}
    query_ids: list[str] = []
    query_texts: list[str] = []
    qrels: dict[str, dict[str, int]] = {}
    with bz2.open(crag_path, "rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if len(query_ids) >= max_queries:
                break
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                continue
            query_text = _extract_query_text(obj)
            if not query_text:
                continue
            raw_qid = str(obj.get("query_id") or obj.get("id") or idx)
            qid = f"crag::q::{raw_qid}"
            rels: dict[str, int] = {}
            for doc in _iter_candidate_docs(obj):
                raw_did = str(doc.get("doc_id") or doc.get("id") or doc.get("url") or f"{idx}-{len(rels)}")
                did = f"crag::doc::{raw_did}"
                text = _extract_doc_text(doc)
                if not text:
                    continue
                if did not in docs and len(docs) >= max_docs:
                    continue
                docs.setdefault(did, text)
                rel = _safe_rel(doc.get("relevance", doc.get("score", 1)))
                rels[did] = max(1, rel)
            if not rels:
                continue
            query_ids.append(qid)
            query_texts.append(query_text)
            qrels[qid] = rels
    doc_ids = list(docs.keys())[:max_docs]
    doc_texts = [docs[doc_id] for doc_id in doc_ids]
    doc_set = set(doc_ids)
    filtered_qrels: dict[str, dict[str, int]] = {}
    for qid in query_ids:
        rels = {did: rel for did, rel in qrels.get(qid, {}).items() if did in doc_set and rel > 0}
        if rels:
            filtered_qrels[qid] = rels
    kept_qids = [qid for qid in query_ids if qid in filtered_qrels][:max_queries]
    kept_qtexts = [query_texts[query_ids.index(qid)] for qid in kept_qids]
    return _TextBundle(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=kept_qids,
        query_texts=kept_qtexts,
        qrels=filtered_qrels,
    )


def _merge_bundles(*, bundles: Sequence[_TextBundle], max_docs: int, max_queries: int) -> _TextBundle:
    docs: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    query_ids: list[str] = []
    query_texts: list[str] = []
    seen_qids: set[str] = set()
    for bundle in bundles:
        for doc_id, doc_text in zip(bundle.doc_ids, bundle.doc_texts):
            if doc_id in docs:
                continue
            if len(docs) >= max_docs:
                break
            docs[doc_id] = doc_text
        for qid, qtext in zip(bundle.query_ids, bundle.query_texts):
            if len(query_ids) >= max_queries:
                break
            key = qid
            suffix = 1
            while key in seen_qids:
                suffix += 1
                key = f"{qid}#{suffix}"
            rels = {did: int(rel) for did, rel in bundle.qrels.get(qid, {}).items() if did in docs and int(rel) > 0}
            if not rels:
                continue
            seen_qids.add(key)
            query_ids.append(key)
            query_texts.append(qtext)
            qrels[key] = rels
    doc_ids = list(docs.keys())[:max_docs]
    doc_texts = [docs[doc_id] for doc_id in doc_ids]
    doc_set = set(doc_ids)
    final_qrels: dict[str, dict[str, int]] = {}
    final_qids: list[str] = []
    final_qtexts: list[str] = []
    for qid, qtext in zip(query_ids, query_texts):
        rels = {did: rel for did, rel in qrels.get(qid, {}).items() if did in doc_set and rel > 0}
        if not rels:
            continue
        final_qids.append(qid)
        final_qtexts.append(qtext)
        final_qrels[qid] = rels
    return _TextBundle(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=final_qids,
        query_texts=final_qtexts,
        qrels=final_qrels,
    )


def _build_retrieval_dataset(*, bundle: _TextBundle, vector_dim: int, seed: int) -> D4RetrievalDataset:
    if not bundle.doc_ids or not bundle.query_ids:
        raise ValueError("D4 bundle must include at least one doc and one query")
    doc_token_sets = [set(tokenize_text(text)) for text in bundle.doc_texts]
    query_token_sets = [set(tokenize_text(text)) for text in bundle.query_texts]
    idf = compute_idf(doc_token_sets)
    doc_vectors = _vectorize_token_sets(doc_token_sets, dim=vector_dim, idf=idf, seed=seed + 11)
    query_vectors = _vectorize_token_sets(query_token_sets, dim=vector_dim, idf=idf, seed=seed + 17)
    return D4RetrievalDataset(
        doc_ids=list(bundle.doc_ids),
        doc_vectors=doc_vectors.astype(np.float32, copy=False),
        doc_texts=list(bundle.doc_texts),
        doc_token_sets=doc_token_sets,
        query_ids=list(bundle.query_ids),
        query_vectors=query_vectors.astype(np.float32, copy=False),
        query_texts=list(bundle.query_texts),
        query_token_sets=query_token_sets,
        qrels={qid: dict(rels) for qid, rels in bundle.qrels.items()},
        idf=idf,
    )


def _vectorize_token_sets(
    token_sets: Sequence[set[str]],
    *,
    dim: int,
    idf: Mapping[str, float],
    seed: int,
) -> np.ndarray:
    vectors = np.zeros((len(token_sets), dim), dtype=np.float32)
    for row, tokens in enumerate(token_sets):
        for token in sorted(tokens):
            idx, sign = _hashed_index_sign(token=token, dim=dim, seed=seed)
            vectors[row, idx] += sign * float(idf.get(token, 1.0))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors /= norms
    return vectors


def _hashed_index_sign(*, token: str, dim: int, seed: int) -> tuple[int, float]:
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    idx = value % dim
    sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
    return int(idx), float(sign)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_candidate_docs(obj: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("search_results", "documents", "docs", "contexts", "passages"):
        value = obj.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
                elif isinstance(item, str):
                    yield {"text": item}
            return
    # Fallback: a single inline doc.
    if any(k in obj for k in ("doc", "document", "context")):
        text = str(obj.get("doc") or obj.get("document") or obj.get("context") or "").strip()
        if text:
            yield {"text": text}


def _extract_query_text(obj: Mapping[str, Any]) -> str:
    for key in ("query", "question", "prompt", "input"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_doc_text(doc: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("title", "text", "contents", "content", "snippet", "body"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return " ".join(chunks).strip()


def _safe_rel(value: Any) -> int:
    try:
        rel = int(float(value))
        return max(rel, 0)
    except Exception:
        return 0
