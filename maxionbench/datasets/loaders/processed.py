"""Loaders for canonical processed dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .d4_synthetic import D4RetrievalDataset, compute_idf, tokenize_text


PROCESSED_SCHEMA_VERSION = "maxionbench-processed-v1"
_LOG = logging.getLogger(__name__)


class SequentialDocIdSequence(Sequence[str]):
    def __init__(self, count: int):
        self._count = int(count)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> str:
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(idx)
        return str(idx)


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
    doc_vectors: np.ndarray | None = None
    query_vectors: np.ndarray | None = None
    embedding_meta: dict[str, Any] | None = None


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
    metric = str(meta.get("metric", "ip")).lower()
    gt_rows = (
        _recompute_ground_truth_ids(
            ids=ids,
            vectors=vectors_np,
            queries=queries_np,
            top_k=top_k,
            metric=metric,
        )
        if int(vectors_np.shape[0]) != int(vectors.shape[0])
        else _resolve_ground_truth_ids(
            gt_ids=gt_ids[: queries_np.shape[0]],
            ids=ids,
            top_k=top_k,
        )
    )
    return ProcessedAnnDataset(
        ids=ids,
        vectors=vectors_np,
        queries=queries_np,
        ground_truth_ids=gt_rows,
        metric=metric,
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
    metric = str(meta.get("metric", "ip")).lower()
    gt_rows = (
        _recompute_filtered_ground_truth_ids(
            ids=ids,
            vectors=vectors_np,
            queries=queries_np,
            top_k=top_k,
            metric=metric,
            query_filters=limited_filters,
            payloads=padded_payloads,
        )
        if int(vectors_np.shape[0]) != int(vectors.shape[0])
        else _resolve_ground_truth_ids(
            gt_ids=gt_ids[: queries_np.shape[0]],
            ids=ids,
            top_k=top_k,
        )
    )

    return ProcessedFilteredAnnDataset(
        ids=ids,
        vectors=vectors_np,
        queries=queries_np,
        ground_truth_ids=gt_rows,
        query_filters=list(limited_filters),
        payloads=padded_payloads,
        metric=metric,
        meta=meta,
    )


def load_processed_d4_bundle(
    root: Path,
    *,
    vector_dim: int,
    seed: int,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    require_precomputed_embeddings: bool = False,
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
            bundles.append(
                _load_processed_text_bundle(
                    subset_dir,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                    require_precomputed_embeddings=require_precomputed_embeddings,
                )
            )
    if include_crag:
        crag_dir = bundle_root / "crag" / crag_slice_name
        if crag_dir.exists():
            bundles.append(
                _load_processed_text_bundle(
                    crag_dir,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                    require_precomputed_embeddings=require_precomputed_embeddings,
                )
            )
    if not bundles:
        raise FileNotFoundError(f"no processed D4 bundles found under {bundle_root}")

    merged = _merge_text_bundles(bundles=bundles, max_docs=max_docs, max_queries=max_queries)
    return _build_retrieval_dataset(bundle=merged, vector_dim=vector_dim, seed=seed)


def load_processed_text_dataset(
    root: Path,
    *,
    vector_dim: int,
    seed: int,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    require_precomputed_embeddings: bool = False,
    max_docs: int = 200_000,
    max_queries: int = 5_000,
    prioritize_qrel_docs: bool = False,
    min_query_retention_ratio: float | None = None,
) -> D4RetrievalDataset:
    dataset_root = _resolve_processed_root(root)
    bundle = _load_processed_text_bundle(
        dataset_root,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        require_precomputed_embeddings=require_precomputed_embeddings,
    )
    merged = _merge_text_bundles(
        bundles=[bundle],
        max_docs=max_docs,
        max_queries=max_queries,
        prioritize_qrel_docs=prioritize_qrel_docs,
    )
    if min_query_retention_ratio is not None:
        if not 0.0 <= float(min_query_retention_ratio) <= 1.0:
            raise ValueError("min_query_retention_ratio must be within [0, 1]")
        raw_query_count = len(bundle.query_ids)
        if raw_query_count > 0 and len(merged.query_ids) < float(min_query_retention_ratio) * float(raw_query_count):
            raise RuntimeError(
                "processed text dataset became too sparse after filtering "
                f"({len(merged.query_ids)}/{raw_query_count} usable queries retained)"
            )
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


def _recompute_ground_truth_ids(
    *,
    ids: Sequence[str],
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    metric: str,
) -> list[list[str]]:
    _LOG.warning(
        "processed dataset vectors were truncated from ground-truth source size; recomputing exact top-%d ids for %d queries",
        int(top_k),
        int(queries.shape[0]),
    )
    return [_exact_topk_ids(ids=ids, vectors=vectors, query=query, top_k=top_k, metric=metric) for query in queries]


def _recompute_filtered_ground_truth_ids(
    *,
    ids: Sequence[str],
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    metric: str,
    query_filters: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    _LOG.warning(
        "processed filtered dataset vectors were truncated from ground-truth source size; recomputing exact filtered top-%d ids for %d queries",
        int(top_k),
        int(queries.shape[0]),
    )
    rows: list[list[str]] = []
    for query, filt in zip(queries, query_filters, strict=False):
        rows.append(
            _exact_topk_ids(
                ids=ids,
                vectors=vectors,
                query=query,
                top_k=top_k,
                metric=metric,
                payloads=payloads,
                filters=filt,
            )
        )
    return rows


def _exact_topk_ids(
    *,
    ids: Sequence[str],
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int,
    metric: str,
    payloads: Sequence[Mapping[str, Any]] | None = None,
    filters: Mapping[str, Any] | None = None,
) -> list[str]:
    normalized_metric = _normalize_metric(metric)
    vectors_np = np.asarray(vectors, dtype=np.float32)
    query_np = np.asarray(query, dtype=np.float32)
    if payloads is not None and filters:
        mask = np.asarray([_matches_processed_filter(payload, filters) for payload in payloads], dtype=bool)
    else:
        mask = np.ones(vectors_np.shape[0], dtype=bool)
    if not np.any(mask):
        return []
    masked_vectors = vectors_np[mask]
    masked_indices = np.flatnonzero(mask)
    if normalized_metric == "cos":
        masked_vectors = _unit_rows(masked_vectors)
        query_np = _unit_vector(query_np)
        scores = masked_vectors @ query_np
        order = np.argsort(-scores, kind="mergesort")
    elif normalized_metric == "l2":
        distances = np.sum((masked_vectors - query_np) ** 2, axis=1)
        order = np.argsort(distances, kind="mergesort")
    else:
        scores = masked_vectors @ query_np
        order = np.argsort(-scores, kind="mergesort")
    selected = masked_indices[order[:top_k]]
    return [str(ids[int(index)]) for index in selected]


def _normalize_metric(metric: str) -> str:
    normalized = str(metric).strip().lower()
    if normalized in {"ip", "inner_product", "dot"}:
        return "ip"
    if normalized in {"l2", "euclid", "euclidean"}:
        return "l2"
    if normalized in {"cos", "cosine", "angular"}:
        return "cos"
    raise ValueError(f"Unsupported processed dataset metric: {metric}")


def _matches_processed_filter(payload: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "must_have_tags":
            payload_tags = payload.get("tags")
            if not isinstance(payload_tags, (list, tuple, set)):
                return False
            expected_tags = list(expected) if isinstance(expected, (list, tuple, set)) else [expected]
            if any(tag not in payload_tags for tag in expected_tags):
                return False
            continue
        if payload.get(key) != expected:
            return False
    return True


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / (float(np.linalg.norm(vector)) + 1e-12)


def _load_processed_text_bundle(
    dataset_dir: Path,
    *,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    require_precomputed_embeddings: bool = False,
) -> _ProcessedTextBundle:
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
    embedding_payload = _load_precomputed_text_embeddings(
        dataset_dir=dataset_dir,
        doc_ids=list(docs.keys()),
        query_ids=kept_qids,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        require_precomputed_embeddings=require_precomputed_embeddings,
    )
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]
    return _ProcessedTextBundle(
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=kept_qids,
        query_texts=kept_qtexts,
        qrels=qrels,
        doc_vectors=embedding_payload["doc_vectors"],
        query_vectors=embedding_payload["query_vectors"],
        embedding_meta=embedding_payload["meta"],
    )


def _merge_text_bundles(
    *,
    bundles: Sequence[_ProcessedTextBundle],
    max_docs: int,
    max_queries: int,
    prioritize_qrel_docs: bool = False,
) -> _ProcessedTextBundle:
    has_precomputed = [bundle.doc_vectors is not None or bundle.query_vectors is not None for bundle in bundles]
    if any(has_precomputed) and not all(has_precomputed):
        raise ValueError("processed text bundle merge encountered mixed precomputed and non-precomputed embeddings")
    docs: dict[str, str] = {}
    doc_vectors: dict[str, np.ndarray] = {}
    qrels: dict[str, dict[str, int]] = {}
    query_ids: list[str] = []
    query_texts: list[str] = []
    query_vectors: list[np.ndarray] = []
    embedding_meta: dict[str, Any] | None = None
    doc_limit_drops = 0
    query_limit_drops = 0
    query_no_rels_drops = 0
    for bundle in bundles:
        if bundle.embedding_meta is not None:
            current_key = (
                str(bundle.embedding_meta.get("model_id") or ""),
                int(bundle.embedding_meta.get("dim") or 0),
            )
            if embedding_meta is None:
                embedding_meta = dict(bundle.embedding_meta)
            else:
                existing_key = (
                    str(embedding_meta.get("model_id") or ""),
                    int(embedding_meta.get("dim") or 0),
                )
                if existing_key != current_key:
                    raise ValueError(
                        "processed text bundle merge encountered mismatched precomputed embeddings: "
                        f"{existing_key} vs {current_key}"
                    )
        priority_doc_ids = _priority_qrel_doc_ids(bundle) if prioritize_qrel_docs else set()
        priority_indices: list[int] = []
        filler_indices: list[int] = []
        for doc_index, doc_id in enumerate(bundle.doc_ids):
            if doc_id in docs:
                continue
            if doc_id in priority_doc_ids:
                priority_indices.append(doc_index)
            else:
                filler_indices.append(doc_index)
        doc_limit_drops += _merge_bundle_doc_indices(
            bundle=bundle,
            indices=priority_indices,
            docs=docs,
            doc_vectors=doc_vectors,
            max_docs=max_docs,
        )
        doc_limit_drops += _merge_bundle_doc_indices(
            bundle=bundle,
            indices=filler_indices,
            docs=docs,
            doc_vectors=doc_vectors,
            max_docs=max_docs,
        )
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
            if bundle.query_vectors is not None:
                query_vectors.append(np.asarray(bundle.query_vectors[query_index], dtype=np.float32))
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
        doc_vectors=(
            np.asarray([doc_vectors[doc_id] for doc_id in docs.keys()], dtype=np.float32)
            if doc_vectors
            else None
        ),
        query_vectors=np.asarray(query_vectors, dtype=np.float32) if query_vectors else None,
        embedding_meta=embedding_meta,
    )


def _priority_qrel_doc_ids(bundle: _ProcessedTextBundle) -> set[str]:
    return {
        str(doc_id)
        for rels in bundle.qrels.values()
        for doc_id, rel in rels.items()
        if int(rel) > 0
    }


def _merge_bundle_doc_indices(
    *,
    bundle: _ProcessedTextBundle,
    indices: Sequence[int],
    docs: dict[str, str],
    doc_vectors: dict[str, np.ndarray],
    max_docs: int,
) -> int:
    for offset, doc_index in enumerate(indices):
        if len(docs) >= max_docs:
            return len(indices) - offset
        doc_id = bundle.doc_ids[doc_index]
        docs[doc_id] = bundle.doc_texts[doc_index]
        if bundle.doc_vectors is not None:
            doc_vectors[doc_id] = np.asarray(bundle.doc_vectors[doc_index], dtype=np.float32)
    return 0


def _build_retrieval_dataset(*, bundle: _ProcessedTextBundle, vector_dim: int, seed: int) -> D4RetrievalDataset:
    if vector_dim < 1:
        raise ValueError("vector_dim must be >= 1")
    if not bundle.doc_ids or not bundle.query_ids:
        raise ValueError("processed text bundle must include at least one doc and one query")
    doc_token_sets = [set(tokenize_text(text)) for text in bundle.doc_texts]
    query_token_sets = [set(tokenize_text(text)) for text in bundle.query_texts]
    idf = compute_idf(doc_token_sets)
    if bundle.doc_vectors is not None or bundle.query_vectors is not None:
        if bundle.doc_vectors is None or bundle.query_vectors is None:
            raise ValueError("processed text bundle must provide both doc_vectors and query_vectors when using precomputed embeddings")
        if bundle.doc_vectors.ndim != 2 or bundle.query_vectors.ndim != 2:
            raise ValueError("precomputed embedding arrays must be 2D")
        if int(bundle.doc_vectors.shape[0]) != len(bundle.doc_ids) or int(bundle.query_vectors.shape[0]) != len(bundle.query_ids):
            raise ValueError("precomputed embedding row counts must match processed text ids")
        if int(bundle.doc_vectors.shape[1]) != int(vector_dim) or int(bundle.query_vectors.shape[1]) != int(vector_dim):
            raise ValueError(
                f"precomputed embedding dimension mismatch: expected {vector_dim}, "
                f"docs={int(bundle.doc_vectors.shape[1])}, queries={int(bundle.query_vectors.shape[1])}"
            )
        doc_vectors = np.asarray(bundle.doc_vectors, dtype=np.float32)
        query_vectors = np.asarray(bundle.query_vectors, dtype=np.float32)
    else:
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


def embedding_model_slug(model_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(model_id).strip().lower()).strip("-")
    if not normalized:
        raise ValueError("embedding model id must not be empty")
    return normalized


def _load_precomputed_text_embeddings(
    *,
    dataset_dir: Path,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    embedding_model: str | None,
    embedding_dim: int | None,
    require_precomputed_embeddings: bool,
) -> dict[str, Any]:
    if not embedding_model:
        if require_precomputed_embeddings:
            raise ValueError("require_precomputed_embeddings=True requires embedding_model")
        return {"doc_vectors": None, "query_vectors": None, "meta": None}

    embedding_dir = dataset_dir / "embeddings" / embedding_model_slug(embedding_model)
    if not embedding_dir.exists():
        if require_precomputed_embeddings:
            raise FileNotFoundError(
                f"missing precomputed embeddings for {embedding_model!r} under {embedding_dir}; "
                "run `maxionbench precompute-text-embeddings` first"
            )
        return {"doc_vectors": None, "query_vectors": None, "meta": None}

    meta_path = embedding_dir / "meta.json"
    doc_vectors_path = embedding_dir / "doc_vectors.npy"
    query_vectors_path = embedding_dir / "query_vectors.npy"
    if not meta_path.exists() or not doc_vectors_path.exists() or not query_vectors_path.exists():
        raise FileNotFoundError(
            f"precomputed embeddings under {embedding_dir} must include meta.json, doc_vectors.npy, and query_vectors.npy"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("model_id") or "") != str(embedding_model):
        raise ValueError(
            f"precomputed embedding model mismatch under {embedding_dir}: "
            f"expected {embedding_model!r}, found {meta.get('model_id')!r}"
        )
    dim = int(meta.get("dim") or 0)
    if embedding_dim is not None and dim != int(embedding_dim):
        raise ValueError(
            f"precomputed embedding dim mismatch under {embedding_dir}: "
            f"expected {embedding_dim}, found {dim}"
        )
    if _ordered_ids_sha256(doc_ids) != str(meta.get("doc_ids_sha256") or ""):
        raise ValueError(f"precomputed doc id checksum mismatch under {embedding_dir}")
    if _ordered_ids_sha256(query_ids) != str(meta.get("query_ids_sha256") or ""):
        raise ValueError(f"precomputed query id checksum mismatch under {embedding_dir}")
    doc_vectors = np.load(doc_vectors_path, mmap_mode="r", allow_pickle=False)
    query_vectors = np.load(query_vectors_path, mmap_mode="r", allow_pickle=False)
    if int(doc_vectors.shape[0]) != len(doc_ids) or int(query_vectors.shape[0]) != len(query_ids):
        raise ValueError(f"precomputed embedding row count mismatch under {embedding_dir}")
    if int(doc_vectors.shape[1]) != dim or int(query_vectors.shape[1]) != dim:
        raise ValueError(f"precomputed embedding column count mismatch under {embedding_dir}")
    return {
        "doc_vectors": np.asarray(doc_vectors, dtype=np.float32),
        "query_vectors": np.asarray(query_vectors, dtype=np.float32),
        "meta": meta,
    }


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
