from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from maxionbench.datasets.loaders.processed import PROCESSED_SCHEMA_VERSION, embedding_model_slug
from maxionbench.tools import precompute_text_embeddings as precompute_mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_dataset(path: Path, *, docs: list[dict], queries: list[dict], qrels: list[tuple[str, str, int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": PROCESSED_SCHEMA_VERSION,
                "task_type": "text_retrieval_strict",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(path / "corpus.jsonl", docs)
    _write_jsonl(path / "queries.jsonl", queries)
    with (path / "qrels.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in qrels:
            handle.write(f"{qid}\t{did}\t{score}\n")


def test_precompute_text_embeddings_writes_recursive_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "processed_d4"
    _write_dataset(
        root / "beir" / "scifact",
        docs=[
            {"doc_id": "scifact::doc::d1", "text": "alpha facts"},
            {"doc_id": "scifact::doc::d2", "text": "beta facts"},
        ],
        queries=[{"query_id": "scifact::q::q1", "text": "alpha facts"}],
        qrels=[("scifact::q::q1", "scifact::doc::d1", 1)],
    )
    _write_dataset(
        root / "beir" / "fiqa",
        docs=[{"doc_id": "fiqa::doc::d1", "text": "bond spreads"}],
        queries=[{"query_id": "fiqa::q::q1", "text": "bond spreads"}],
        qrels=[("fiqa::q::q1", "fiqa::doc::d1", 1)],
    )

    def _fake_load_text_encoder(*, model_id: str, batch_size: int, device: str, max_length: int, normalize: bool):  # type: ignore[no-untyped-def]
        del batch_size, device, max_length

        def _encode(texts: list[str] | tuple[str, ...]) -> np.ndarray:
            rows = []
            for idx, text in enumerate(texts):
                token_score = float(sum(ord(ch) for ch in text) % 17)
                rows.append([float(idx + 1), token_score, float(len(text.split())), 1.0 if normalize else 2.0])
            return np.asarray(rows, dtype=np.float32)

        return precompute_mod._TextEncoder(
            model_id=model_id,
            device="cpu",
            dim=4,
            pooling="mean",
            normalize=normalize,
            encode=_encode,
        )

    monkeypatch.setattr(precompute_mod, "_load_text_encoder", _fake_load_text_encoder)
    summary = precompute_mod.precompute_text_embeddings(
        input_path=root,
        model_id="BAAI/bge-small-en-v1.5",
        batch_size=8,
        device="cpu",
        max_length=128,
        normalize=True,
        force=False,
    )

    assert summary["datasets_found"] == 2
    assert summary["datasets_processed"] == 2
    embedding_dir = root / "beir" / "scifact" / "embeddings" / embedding_model_slug("BAAI/bge-small-en-v1.5")
    assert (embedding_dir / "doc_vectors.npy").exists()
    assert (embedding_dir / "query_vectors.npy").exists()
    meta = json.loads((embedding_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_id"] == "BAAI/bge-small-en-v1.5"
    assert meta["dim"] == 4
    doc_vectors = np.load(embedding_dir / "doc_vectors.npy", allow_pickle=False)
    query_vectors = np.load(embedding_dir / "query_vectors.npy", allow_pickle=False)
    assert doc_vectors.shape == (2, 4)
    assert query_vectors.shape == (1, 4)

    cached = precompute_mod.precompute_text_embeddings(
        input_path=root,
        model_id="BAAI/bge-small-en-v1.5",
        batch_size=8,
        device="cpu",
        max_length=128,
        normalize=True,
        force=False,
    )
    assert cached["datasets_processed"] == 0
    assert cached["datasets_skipped"] == 2
