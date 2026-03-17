"""Loaders for canonical processed dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from maxionbench.datasets.d3_generator import SequentialDocIdSequence

from .d4_synthetic import D4RetrievalDataset, compute_idf, tokenize_text


PROCESSED_SCHEMA_VERSION = "maxionbench-processed-v1"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedAnnDataset:
    ids: Sequence[str]
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: list[list[str]]
    metric: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class ProcessedFilteredAnnDataset:
    ids: Sequence[str]
    vectors: np.ndarray
    queries: np.ndarray
    ground_truth_ids: list[list[str]]
    query_filters: Sequence[dict[str, Any]]
    payloads: Sequence[dict[str, Any]]
    metric: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class _ProcessedTextBundle:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]


def load_processed_ann_dataset(
    path: Path,
    *,
    max_vectors: int | None = None,
    max_queries: int | None = None,
    top_k: int = 10,
) -> ProcessedAnnDataset:
    root = _resolve_processed_root(path)
    meta = _load_meta(root)
    task_type = str(meta.get("task_type", "")).strip().lower()
    if task_type not in {"ann", "filtered_ann"}:
        raise ValueError(f"processed ANN dataset must declare task_type ann or filtered_ann, got {task_type!r}")

    vectors = _load_required_array(root / "base.npy", mmap=True)
    queries = _load_required_array(root / "queries.npy", mmap=True)
    gt_ids = _load_required_array(root / "gt_ids.npy", mmap=True)

    vectors_np = np.asarray(vectors[:max_vectors] if max_vectors is not None else vectors).astype(np.float32, copy=False)
    queries_np = np.asarray(queries[:max_queries] if max_queries is not None else queries).astype(np.float32, copy=False)
    ids = SequentialDocIdSequence(int(vectors_np.shape[0]))
    gt_rows = _resolve_ground_truth_ids(
        gt_ids=gt_ids[: queries_np.shape[0]],
        ids=ids,
        top_k=top_k,
    )
    return ProcessedAnnDataset(
        ids=ids,
        vectors=vectors_np,
        queries=queries_np,
        ground_truth_ids=gt_rows,
        metric=str(meta.get("metric", "ip")).lower(),
        meta=meta,
    )


def load_processed_filtered_ann_dataset(
    path: Path,
    *,
    max_vectors: int | None = None,
    max_queries: int | None = None,
    top_k: int = 10,
) -> ProcessedFilteredAnnDataset:
    root = _resolve_processed_root(path)
    meta = _load_meta(root)
    task_type = str(meta.get("task_type", "")).strip().lower()
    if task_type != "filtered_ann":
        raise ValueError(f"processed filtered ANN dataset must declare task_type filtered_ann, got {task_type!r}")

    vectors = _load_required_array(root / "base.npy", mmap=True)
    queries = _load_required_array(root / "queries.npy", mmap=True)
    gt_ids = _load_required_array(root / "gt_ids.npy", mmap=True)
    filters = list(_read_jsonl(root / "filters.jsonl"))
    payload_path = root / "payloads.jsonl"
    payloads = list(_read_jsonl(payload_path)) if payload_path.exists() else []

    vectors_np = np.asarray(vectors[:max_vectors] if max_vectors is not None else vectors).astype(np.float32, copy=False)
    queries_np = np.asarray(queries[:max_queries] if max_queries is not None else queries).astype(np.float32, copy=False)
    ids = SequentialDocIdSequence(int(vectors_np.shape[0]))
    gt_rows = _resolve_ground_truth_ids(
        gt_ids=gt_ids[: queries_np.shape[0]],
        ids=ids,
        top_k=top_k,
    )

    limited_filters = filters[: queries_np.shape[0]]
    if len(limited_filters) < int(queries_np.shape[0]):
        raise ValueError("processed filtered ANN dataset does not include one filter row per query")
    if payloads and len(payloads) < len(ids):
        raise ValueError("processed filtered ANN payload count is smaller than vector count")
    if payloads and len(payloads) > len(ids):
        _LOG.warning(
            "processed filtered ANN dataset payloads.jsonl has %d rows for %d ids; truncating extras",
            len(payloads),
            len(ids),
        )
    padded_payloads = payloads[: len(ids)] if payloads else [{} for _ in range(len(ids))]

    return ProcessedFilteredAnnDataset(
        ids=ids,
        vectors=vectors_np,
        queries=queries_np,
        ground_truth_ids=gt_rows,
        query_filters=list(limited_filters),
        payloads=padded_payloads,
        metric=str(meta.get("metric", "ip")).lower(),
        meta=meta,
    )


def load_processed_d4_bundle(
    root: Path,
    *,
    vector_dim: int,
    seed: int,
    beir_subsets: Sequence[str] | None = None,
    include_crag: bool = True,
    crag_slice_name: str = "small_slice",
    max_docs: int = 200_000,
    max_queries: int = 5_000,
) -> D4RetrievalDataset:
    bundle_root = _resolve_processed_root(root)
    bundles: list[_ProcessedTextBundle] = []

    for subset in list(beir_subsets or []):
        subset_dir = bundle_root / "beir" / subset
        if subset_dir.exists():
            bundles.append(_load_processed_text_bundle(subset_dir))
    if include_crag:
        crag_dir = bundle_root / "crag" / crag_slice_name
        if crag_dir.exists():
            bundles.append(_load_processed_text_bundle(crag_dir))
    if not bundles:
        raise FileNotFoundError(f"no processed D4 bundles found under {bundle_root}")

    merged = _merge_text_bundles(bundles=bundles, max_docs=max_docs, max_queries=max_queries)
    return _build_retrieval_dataset(bundle=merged, vector_dim=vector_dim, seed=seed)


def dataset_dir_sha256(path: Path) -> str:
    root = _resolve_processed_root(path)
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_processed_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"processed dataset path not found: {root}")
    return root


def _load_meta(root: Path) -> dict[str, Any]:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"processed dataset is missing meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"processed dataset meta.json must be an object: {meta_path}")
    schema_version = str(meta.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError(
            f"processed dataset meta.json must declare schema_version={PROCESSED_SCHEMA_VERSION!r}: {meta_path}"
        )
    if schema_version != PROCESSED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported processed dataset schema_version {schema_version!r}; expected {PROCESSED_SCHEMA_VERSION!r}"
        )
    return meta


def _load_required_array(path: Path, *, mmap: bool) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _resolve_ground_truth_ids(
    *,
    gt_ids: np.ndarray,
    ids: Sequence[str],
    top_k: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    array = np.asarray(gt_ids)
    if array.ndim != 2:
        raise ValueError(f"gt_ids.npy must be 2D [Q, K]; got shape={tuple(array.shape)}")
    if array.dtype.kind in {"U", "S", "O"}:
        for row in array[:, :top_k]:
            rows.append([str(item) for item in row])
        return rows
    indexed = np.asarray(array, dtype=np.int64)
    invalid_count = int(np.count_nonzero((indexed[:, :top_k] < 0) | (indexed[:, :top_k] >= len(ids))))
    if invalid_count:
        _LOG.warning(
            "processed dataset contains %d out-of-bounds ground-truth indices (valid id range: 0..%d)",
            invalid_count,
            max(len(ids) - 1, 0),
        )
        raise ValueError("processed dataset gt_ids.npy contains out-of-bounds indices")
    for row in indexed[:, :top_k]:
        rows.append([ids[int(idx)] for idx in row])
    return rows


def _load_processed_text_bundle(dataset_dir: Path) -> _ProcessedTextBundle:
    meta = _load_meta(dataset_dir)
    task_type = str(meta.get("task_type", "")).strip().lower()
    if task_type not in {"text_retrieval", "text_retrieval_strict", "text_retrieval_weak", "text_retrieval_weak_labels"}:
        raise ValueError(f"processed text dataset must declare a text_retrieval* task_type, got {task_type!r}")

    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels.tsv"
    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(f"processed text dataset is missing corpus.jsonl / queries.jsonl / qrels.tsv under {dataset_dir}")

    docs: dict[str, str] = {}
    for row in _read_jsonl(corpus_path):
        doc_id = str(row.get("doc_id") or row.get("_id") or row.get("id") or "").strip()
        if not doc_id:
            continue
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        full_text = f"{title} {text}".strip()
        if full_text:
            docs[doc_id] = full_text
        else:
            _LOG.warning(
                "processed text dataset %s skipped empty doc %s",
                dataset_dir,
                doc_id,
            )

    queries: dict[str, str] = {}
    for row in _read_jsonl(queries_path):
        query_id = str(row.get("query_id") or row.get("_id") or row.get("id") or "").strip()
        text = str(row.get("text") or row.get("query") or "").strip()
        if query_id and text:
            queries[query_id] = text

    qrels = _read_qrels(qrels_path=qrels_path, allowed_qids=set(queries.keys()), allowed_doc_ids=set(docs.keys()))
    kept_qids = [qid for qid in queries.keys() if qid in qrels]
    kept_qtexts = [queries[qid] for qid in kept_qids]
    dropped_queries = len(queries) - len(kept_qids)
    if dropped_queries:
        _LOG.warning(
            "processed text dataset %s dropped %d queries without surviving qrels after filtering",
            dataset_dir,
            dropped_queries,
        )
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]
    return _ProcessedTextBundle(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=kept_qids,
        query_texts=kept_qtexts,
        qrels=qrels,
    )


def _merge_text_bundles(
    *,
    bundles: Sequence[_ProcessedTextBundle],
    max_docs: int,
    max_queries: int,
) -> _ProcessedTextBundle:
    docs: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    query_ids: list[str] = []
    query_texts: list[str] = []
    doc_limit_drops = 0
    query_limit_drops = 0
    query_no_rels_drops = 0
    for bundle in bundles:
        for doc_index, (doc_id, doc_text) in enumerate(zip(bundle.doc_ids, bundle.doc_texts)):
            if doc_id in docs:
                continue
            if len(docs) >= max_docs:
                doc_limit_drops += len(bundle.doc_ids) - doc_index
                break
            docs[doc_id] = doc_text
        for query_index, (qid, qtext) in enumerate(zip(bundle.query_ids, bundle.query_texts)):
            if len(query_ids) >= max_queries:
                query_limit_drops += len(bundle.query_ids) - query_index
                break
            rels = {did: int(rel) for did, rel in bundle.qrels.get(qid, {}).items() if did in docs and int(rel) > 0}
            if not rels:
                query_no_rels_drops += 1
                continue
            query_ids.append(qid)
            query_texts.append(qtext)
            qrels[qid] = rels
    if doc_limit_drops or query_limit_drops or query_no_rels_drops:
        _LOG.warning(
            "processed D4 merge dropped docs/queries during bundle merge (doc_limit_drops=%d, query_limit_drops=%d, query_missing_qrels=%d)",
            doc_limit_drops,
            query_limit_drops,
            query_no_rels_drops,
        )
    if not query_ids:
        raise ValueError(
            "processed D4 merge produced 0 queries after filtering "
            f"(doc_limit_drops={doc_limit_drops}, query_limit_drops={query_limit_drops}, "
            f"query_missing_qrels={query_no_rels_drops})"
        )
    return _ProcessedTextBundle(
        doc_ids=list(docs.keys()),
        doc_texts=[docs[doc_id] for doc_id in docs.keys()],
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )


def _build_retrieval_dataset(*, bundle: _ProcessedTextBundle, vector_dim: int, seed: int) -> D4RetrievalDataset:
    if vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    if not bundle.doc_ids or not bundle.query_ids:
        raise ValueError("processed text bundle must include at least one doc and one query")
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
            obj = json.loads(row)
            if not isinstance(obj, dict):
                continue
            yield obj


def _read_qrels(
    *,
    qrels_path: Path,
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
            if line_no == 0 and any("query" in part.lower() for part in parts):
                continue
            if len(parts) < 3:
                continue
            qid = parts[0].strip()
            did = parts[1].strip()
            if qid not in allowed_qids or did not in allowed_doc_ids:
                continue
            rel = _safe_int(parts[2])
            if rel <= 0:
                continue
            qrels.setdefault(qid, {})[did] = max(rel, qrels.get(qid, {}).get(did, 0))
    return qrels


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0
