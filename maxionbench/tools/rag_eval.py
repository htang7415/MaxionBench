"""v0.2 CPU RAG evaluation: per-stage retrieval quality/latency, then end-to-end generation.

Retrieval pipelines: dense (exact inner product over precomputed embeddings; same ranking as FAISS FlatIP), BM25, hybrid (RRF of
dense + BM25), and hybrid + cross-encoder rerank. Generation sends the top-k context to an
OpenAI-compatible endpoint and records answer EM/F1 with measured TTFT, latency, and tokens.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable
import urllib.request

import numpy as np

from maxionbench.datasets.loaders.processed import embedding_model_slug
from maxionbench.metrics.latency import latency_summary
from maxionbench.rag.answer_metrics import exact_match, token_f1
from maxionbench.rag.fusion import reciprocal_rank_fusion
from maxionbench.rag.llm_client import chat_completion
from maxionbench.rag.stats import paired_generation_deltas
from maxionbench.tools.precompute_text_embeddings import _load_text_dataset_rows, _read_qrels

CANDIDATE_DEPTH = 100
SYSTEM_PROMPT = (
    "Answer the question using only the numbered documents. Reply with the shortest possible answer "
    "phrase and no explanation. For yes/no questions reply yes or no."
)


def evidence_coverage(retrieved: list[str], gold: set[str]) -> float:
    return len(set(retrieved) & gold) / len(gold) if gold else 0.0


def build_messages(question: str, docs: list[str]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(docs, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]


def _top_k_inner_product(matrix: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    scores = matrix @ query
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def _log(msg: str) -> None:
    print(f"[rag-eval] {msg}", file=sys.stderr, flush=True)


def run(args: Any) -> dict[str, Any]:
    import bm25s  # type: ignore[import-not-found]
    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

    dataset_dir = Path(args.dataset)
    rows = _load_text_dataset_rows(dataset_dir)
    qrels = _read_qrels(
        qrels_path=dataset_dir / "qrels.tsv", allowed_qids=set(rows.query_ids), allowed_doc_ids=set(rows.doc_ids)
    )
    answers = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            answers[row["query_id"]] = row.get("answer") or ""

    emb_dir = dataset_dir / "embeddings" / embedding_model_slug(args.embedding_model)
    doc_vecs = np.load(emb_dir / "doc_vectors.npy")
    query_vecs = np.load(emb_dir / "query_vectors.npy")
    if doc_vecs.shape[0] != len(rows.doc_ids) or query_vecs.shape[0] != len(rows.query_ids):
        raise ValueError(f"embedding rows do not match dataset rows under {emb_dir}")

    rng = random.Random(args.seed)
    sample = sorted(rng.sample(range(len(rows.query_ids)), min(args.retrieval_queries, len(rows.query_ids))))

    # Exact dense search in numpy rather than faiss: faiss-cpu and torch each bundle libomp on
    # macOS, and loading both in one process deadlocks or segfaults.
    doc_matrix, dense_build_s = _timed(lambda: np.ascontiguousarray(doc_vecs, dtype=np.float32))
    bm25 = bm25s.BM25()
    _, bm25_build_s = _timed(lambda: bm25.index(bm25s.tokenize(rows.doc_texts, stopwords="en", show_progress=False), show_progress=False))
    reranker = CrossEncoder(args.reranker, device="cpu")
    _log(f"indexes built: dense {dense_build_s:.1f}s bm25 {bm25_build_s:.1f}s; {len(sample)} queries")

    doc_pos = {d: i for i, d in enumerate(rows.doc_ids)}
    sample_pos = {qi: pos for pos, qi in enumerate(sample)}
    pipelines: dict[int, dict[str, list[str]]] = {}
    stage_s: dict[str, list[float]] = {"dense": [], "bm25": [], "rrf": [], "rerank": []}
    for n, qi in enumerate(sample, start=1):
        qtext = rows.query_texts[qi]
        dense_idx, t_dense = _timed(lambda: _top_k_inner_product(doc_matrix, query_vecs[qi], CANDIDATE_DEPTH))
        dense = [rows.doc_ids[j] for j in dense_idx]
        (bm_idx, _), t_bm25 = _timed(
            lambda: bm25.retrieve(bm25s.tokenize([qtext], stopwords="en", show_progress=False), k=CANDIDATE_DEPTH, show_progress=False)
        )
        sparse = [rows.doc_ids[j] for j in bm_idx[0]]
        hybrid, t_rrf = _timed(lambda: reciprocal_rank_fusion([dense, sparse], top_k=CANDIDATE_DEPTH))
        pool = hybrid[: args.rerank_depth]
        scores, t_rerank = _timed(lambda: reranker.predict([(qtext, rows.doc_texts[doc_pos[d]]) for d in pool], batch_size=32))
        reranked = [pool[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
        pipelines[qi] = {"dense": dense, "bm25": sparse, "hybrid": hybrid, "hybrid_rerank": reranked}
        for name, t in (("dense", t_dense), ("bm25", t_bm25), ("rrf", t_rrf), ("rerank", t_rerank)):
            stage_s[name].append(t)
        if n % 200 == 0:
            _log(f"retrieval {n}/{len(sample)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_stages = {
        "dense": ["dense"],
        "bm25": ["bm25"],
        "hybrid": ["dense", "bm25", "rrf"],
        "hybrid_rerank": ["dense", "bm25", "rrf", "rerank"],
    }
    retrieval_summary: dict[str, Any] = {}
    with (out_dir / "retrieval.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "pipeline", "coverage_at_3", "coverage_at_10", "latency_ms"])
        for name, stages in pipeline_stages.items():
            cov3, cov10, lat = [], [], []
            for pos, qi in enumerate(sample):
                gold = set(qrels[rows.query_ids[qi]])
                ranked = pipelines[qi][name]
                ms = 1000 * sum(stage_s[s][pos] for s in stages)
                cov3.append(evidence_coverage(ranked[:3], gold))
                cov10.append(evidence_coverage(ranked[:10], gold))
                lat.append(ms)
                writer.writerow([rows.query_ids[qi], name, f"{cov3[-1]:.4f}", f"{cov10[-1]:.4f}", f"{ms:.3f}"])
            retrieval_summary[name] = {
                "evidence_coverage_at_3": round(float(np.mean(cov3)), 4),
                "evidence_coverage_at_10": round(float(np.mean(cov10)), 4),
                "latency": {k: round(v, 2) for k, v in latency_summary(lat).items()},
            }

    generation_summary: dict[str, Any] = {}
    all_recs: list[dict[str, Any]] = []
    if args.gen_queries > 0:
        gen_sample = sample[: args.gen_queries]
        ks = [int(k) for k in args.ks.split(",")]
        with (out_dir / "generation.jsonl").open("w", encoding="utf-8") as fh:
            for name in args.gen_pipelines.split(","):
                for k in ks:
                    recs = []
                    for qi in gen_sample:
                        qid = rows.query_ids[qi]
                        top = pipelines[qi][name][:k]
                        msgs = build_messages(rows.query_texts[qi], [rows.doc_texts[doc_pos[d]] for d in top])
                        res = chat_completion(
                            args.llm_url, msgs, max_tokens=args.max_tokens, timeout_s=args.timeout_s,
                            extra_body={"cache_prompt": False},
                        )
                        retr_ms = 1000 * sum(stage_s[s][sample_pos[qi]] for s in pipeline_stages[name])
                        rec = {
                            "query_id": qid, "pipeline": name, "k": k, "status": res.status,
                            "answer": res.text.strip(), "gold": answers[qid],
                            "em": exact_match(res.text, answers[qid]), "f1": token_f1(res.text, answers[qid]),
                            "coverage": evidence_coverage(top, set(qrels[qid])),
                            "retrieval_ms": round(retr_ms, 3),
                            "ttft_ms": None if res.ttft_s is None else round(1000 * res.ttft_s, 1),
                            "llm_e2e_ms": round(1000 * res.e2e_s, 1),
                            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        recs.append(rec)
                        all_recs.append(rec)
                    ok = [r for r in recs if r["status"] == "ok"]
                    total_s = sum(r["retrieval_ms"] + r["llm_e2e_ms"] for r in ok) / 1000
                    correct = sum(r["em"] for r in ok)
                    generation_summary[f"{name}@k{k}"] = {
                        "n": len(recs), "ok": len(ok),
                        "em": round(float(np.mean([r["em"] for r in ok])), 4) if ok else 0.0,
                        "f1": round(float(np.mean([r["f1"] for r in ok])), 4) if ok else 0.0,
                        "context_coverage": round(float(np.mean([r["coverage"] for r in ok])), 4) if ok else 0.0,
                        "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in ok])), 1) if ok else 0.0,
                        "ttft": {k2: round(v, 1) for k2, v in latency_summary([r["ttft_ms"] for r in ok if r["ttft_ms"]]).items()} if ok else {},
                        "retrieval_share_of_e2e": round(sum(r["retrieval_ms"] for r in ok) / 1000 / total_s, 4) if total_s else 0.0,
                        "serial_seconds_per_correct_answer": round(total_s / correct, 2) if correct else None,
                    }
                    _log(f"generation {name}@k{k}: {generation_summary[f'{name}@k{k}']}")

    summary = {
        "profile": "maxionbench-v0.2-cpu",
        "dataset": str(dataset_dir),
        "embedding_model": args.embedding_model,
        "reranker": args.reranker,
        "rerank_depth": args.rerank_depth,
        "seed": args.seed,
        "retrieval_queries": len(sample),
        "index_build_s": {"dense_exact": round(dense_build_s, 2), "bm25": round(bm25_build_s, 2)},
        "retrieval": retrieval_summary,
        "generation": generation_summary,
        "paired": paired_generation_deltas(all_recs, _comparisons(args)),
        "llm_endpoint": _endpoint_info(args.llm_url) if args.gen_queries > 0 else None,
        "max_tokens": args.max_tokens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _comparisons(args: Any) -> list[tuple[str, str]]:
    """Each pipeline vs the first at equal k, and each pipeline's largest k vs its smallest."""
    pipelines = args.gen_pipelines.split(",")
    ks = sorted(int(k) for k in args.ks.split(","))
    pairs = [(f"{p}@k{k}", f"{pipelines[0]}@k{k}") for k in ks for p in pipelines[1:]]
    pairs += [(f"{p}@k{ks[-1]}", f"{p}@k{ks[0]}") for p in pipelines if len(ks) > 1]
    return pairs


def recompute_paired(args: Any) -> dict[str, Any]:
    """Recompute paired deltas from an existing generation.jsonl without re-running the LLM."""
    out_dir = Path(args.out)
    with (out_dir / "generation.jsonl").open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    summary["paired"] = paired_generation_deltas(records, _comparisons(args))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary["paired"]


def _endpoint_info(base_url: str) -> Any:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as resp:
            return json.load(resp)
    except OSError as exc:
        return {"error": str(exc)}


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/processed/hotpot_portable")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-depth", type=int, default=30)
    parser.add_argument("--retrieval-queries", type=int, default=1000)
    parser.add_argument("--gen-queries", type=int, default=80)
    parser.add_argument("--gen-pipelines", default="dense,hybrid_rerank")
    parser.add_argument("--ks", default="3,10")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8091")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/v0.2/rag_eval")
    parser.add_argument("--recompute-paired", action="store_true", help="only recompute paired deltas in --out")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(recompute_paired(args) if args.recompute_paired else run(args), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
