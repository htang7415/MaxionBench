# NeurIPS Action Plan

## Submission target

Target the NeurIPS Evaluations & Datasets track unless the paper is reframed as a narrowly scoped main-track systems analysis.

## Claim to defend

MaxionBench-Portable is a reproducible, conformance-gated benchmark for agentic retrieval infrastructure on commodity hardware. It shows that strict-latency deployment choices can be benchmarked locally in one day, while full ranking stability is workload-dependent and should be reported rather than assumed.

Do not claim that all deployment decisions are stable. Current S1 and S2 top-1 choices change from B0 to B2.

## Result story

- Strict 200 ms p99 deployment: FAISS CPU with bge-small is the minimum viable choice for S1, S2, and S3.
- No-p99 cost objective: S3 shifts to LanceDB in-process with bge-small.
- Quality-first objective: S1 shifts to LanceDB in-process with bge-base, S2 to FAISS CPU with bge-base, and S3 to pgvector with bge-base.
- Budget stability is mixed: S1/S2 top-1 changes; S3 top-1 is stable despite low full-rank Spearman correlation.

## Deadline-safe status

Do not start broad reruns unless the submission deadline moves. The current paper has enough evidence for the decision-centered story if the claims stay scoped.

Completed de-risking checks:

- S3 matched-query audit over 5,000 HotpotQA-portable queries: pgvector does not show a substantive quality advantage over FAISS-base.
- S2 standard Qdrant HNSW64 competitor row: completed with result parquet and observation log.
- S2 paired FAISS/Qdrant analysis: Qdrant minus FAISS nDCG@10 = -0.002780 with 95% CI [-0.007268, 0.000574], equal freshness over 500 matched events.
- S2 same-orchestration deadline mini-bundle: completed FAISS CPU and Qdrant with two repeats per engine, zero errors, Qdrant minus FAISS nDCG@10 = -0.000734 with 95% CI [-0.002609, 0.000484], and equal freshness over 80 matched events. This is repeatability evidence, not replacement latency evidence, because the run caps timed phases.
- Artifact/reproducibility pass: archive metadata, hardware/runtime, dataset manifest/checksum pointers, and artifact README are now documented.

## Remaining best improvement if time appears

1. Run a second hardware-profile check for strict winners and nearest alternatives.
2. If more time is available, run the same-orchestration S2 mini-bundle without deadline caps.
3. Otherwise stop experiments and polish writing/layout only.
