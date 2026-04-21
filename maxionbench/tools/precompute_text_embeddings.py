"""Precompute model-backed embeddings for processed text datasets."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION, embedding_model_slug

_EMBEDDING_SCHEMA_VERSION = "maxionbench-text-embeddings-v1"
_PRECOMPUTE_VERSION = "0.1"


@dataclass(frozen=True)
class _TextDatasetRows:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]


@dataclass(frozen=True)
class _TextEncoder:
    model_id: str
    device: str
    dim: int
    pooling: str
    normalize: bool
    encode: Callable[[Sequence[str]], np.ndarray]


def precompute_text_embeddings(
    *,
    input_path: Path,
    model_id: str,
    batch_size: int = 16,
    device: str = "auto",
    max_length: int = 512,
    normalize: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    dataset_dirs = _discover_processed_text_datasets(input_path)
    if not dataset_dirs:
        raise FileNotFoundError(f"no processed text datasets found under {input_path}")

    encoder = _load_text_encoder(
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        normalize=normalize,
    )

    outputs: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        rows = _load_text_dataset_rows(dataset_dir)
        outputs.append(
            _write_embedding_artifacts(
                dataset_dir=dataset_dir,
                rows=rows,
                encoder=encoder,
                batch_size=batch_size,
                max_length=max_length,
                force=force,
            )
        )

    processed = sum(1 for row in outputs if row["status"] == "processed")
    skipped = sum(1 for row in outputs if row["status"] == "cache_hit")
    return {
        "input_path": str(input_path.expanduser().resolve()),
        "model_id": model_id,
        "device": encoder.device,
        "dim": encoder.dim,
        "datasets_found": len(dataset_dirs),
        "datasets_processed": processed,
        "datasets_skipped": skipped,
        "outputs": outputs,
    }


def _write_embedding_artifacts(
    *,
    dataset_dir: Path,
    rows: _TextDatasetRows,
    encoder: _TextEncoder,
    batch_size: int,
    max_length: int,
    force: bool,
) -> dict[str, Any]:
    embedding_dir = dataset_dir / "embeddings" / embedding_model_slug(encoder.model_id)
    meta_path = embedding_dir / "meta.json"
    doc_vectors_path = embedding_dir / "doc_vectors.npy"
    query_vectors_path = embedding_dir / "query_vectors.npy"
    doc_ids_sha256 = _ordered_ids_sha256(rows.doc_ids)
    query_ids_sha256 = _ordered_ids_sha256(rows.query_ids)

    if not force and meta_path.exists() and doc_vectors_path.exists() and query_vectors_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            str(existing_meta.get("model_id") or "") == encoder.model_id
            and int(existing_meta.get("dim") or 0) == encoder.dim
            and str(existing_meta.get("doc_ids_sha256") or "") == doc_ids_sha256
            and str(existing_meta.get("query_ids_sha256") or "") == query_ids_sha256
        ):
            return {
                "dataset_dir": str(dataset_dir.resolve()),
                "embedding_dir": str(embedding_dir.resolve()),
                "status": "cache_hit",
                "doc_count": len(rows.doc_ids),
                "query_count": len(rows.query_ids),
                "dim": encoder.dim,
            }

    doc_vectors = encoder.encode(rows.doc_texts)
    query_vectors = encoder.encode(rows.query_texts)
    if doc_vectors.shape != (len(rows.doc_ids), encoder.dim):
        raise ValueError(f"doc embedding shape mismatch for {dataset_dir}: {doc_vectors.shape}")
    if query_vectors.shape != (len(rows.query_ids), encoder.dim):
        raise ValueError(f"query embedding shape mismatch for {dataset_dir}: {query_vectors.shape}")

    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(doc_vectors_path, np.asarray(doc_vectors, dtype=np.float32))
    np.save(query_vectors_path, np.asarray(query_vectors, dtype=np.float32))
    meta = {
        "schema_version": _EMBEDDING_SCHEMA_VERSION,
        "precompute_version": _PRECOMPUTE_VERSION,
        "processed_dataset_schema_version": PROCESSED_SCHEMA_VERSION,
        "model_id": encoder.model_id,
        "dim": encoder.dim,
        "device": encoder.device,
        "pooling": encoder.pooling,
        "normalize": encoder.normalize,
        "batch_size": batch_size,
        "max_length": max_length,
        "doc_count": len(rows.doc_ids),
        "query_count": len(rows.query_ids),
        "doc_ids_sha256": doc_ids_sha256,
        "query_ids_sha256": query_ids_sha256,
        "created_at_utc": _utc_now_iso(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "embedding_dir": str(embedding_dir.resolve()),
        "status": "processed",
        "doc_count": len(rows.doc_ids),
        "query_count": len(rows.query_ids),
        "dim": encoder.dim,
    }


def _load_text_dataset_rows(dataset_dir: Path) -> _TextDatasetRows:
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

    queries: dict[str, str] = {}
    for row in _read_jsonl(queries_path):
        query_id = str(row.get("query_id") or row.get("_id") or row.get("id") or "").strip()
        text = str(row.get("text") or row.get("query") or "").strip()
        if query_id and text:
            queries[query_id] = text

    qrels = _read_qrels(qrels_path=qrels_path, allowed_qids=set(queries.keys()), allowed_doc_ids=set(docs.keys()))
    kept_qids = [qid for qid in queries.keys() if qid in qrels]
    return _TextDatasetRows(
        doc_ids=list(docs.keys()),
        doc_texts=[docs[doc_id] for doc_id in docs.keys()],
        query_ids=kept_qids,
        query_texts=[queries[qid] for qid in kept_qids],
    )


def _discover_processed_text_datasets(root: Path) -> list[Path]:
    resolved = root.expanduser().resolve()
    if _is_processed_text_dataset_dir(resolved):
        return [resolved]
    return sorted(path for path in resolved.rglob("*") if path.is_dir() and _is_processed_text_dataset_dir(path))


def _is_processed_text_dataset_dir(path: Path) -> bool:
    return (
        (path / "meta.json").is_file()
        and (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and (path / "qrels.tsv").is_file()
    )


def _load_text_encoder(
    *,
    model_id: str,
    batch_size: int,
    device: str,
    max_length: int,
    normalize: bool,
) -> _TextEncoder:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import path exercised by users, not CI
        raise RuntimeError(
            "precompute-text-embeddings requires `torch` and `transformers`; "
            'install MaxionBench with the embeddings extra (`python -m pip install -e ".[embeddings]"`).'
        ) from exc

    resolved_device = _resolve_torch_device(torch=torch, requested=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()
    dim = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model", 0) or 0)
    if dim < 1:
        raise ValueError(f"unable to determine embedding dimension for {model_id}")

    def _encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state
                attention = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(rows, axis=0)

    return _TextEncoder(
        model_id=model_id,
        device=str(resolved_device),
        dim=dim,
        pooling="mean",
        normalize=normalize,
        encode=_encode,
    )


def _resolve_torch_device(*, torch: Any, requested: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized and normalized != "auto":
        return normalized
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    digest = json.dumps(list(ids), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(digest).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            payload = json.loads(row)
            if isinstance(payload, dict):
                yield payload


def _read_qrels(*, qrels_path: Path, allowed_qids: set[str], allowed_doc_ids: set[str]) -> dict[str, dict[str, int]]:
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
            doc_id = parts[1].strip()
            if qid not in allowed_qids or doc_id not in allowed_doc_ids:
                continue
            try:
                rel = max(0, int(float(parts[2])))
            except Exception:
                rel = 0
            if rel > 0:
                qrels.setdefault(qid, {})[doc_id] = max(rel, qrels.get(qid, {}).get(doc_id, 0))
    return qrels


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description="Precompute model-backed embeddings for processed text datasets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = precompute_text_embeddings(
        input_path=Path(args.input),
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        normalize=not bool(args.no_normalize),
        force=bool(args.force),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
