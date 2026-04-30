# NeurIPS readiness plan

This plan follows the `.agents` paper-writing guidance: every strong claim needs explicit evidence, and unsupported claims should become either new experiments or limitations.

## Current strongest story

MaxionBench-Portable is best framed as a decision-centered benchmark for agentic retrieval infrastructure. The paper's strongest contribution is not an engine leaderboard; it is the protocol that combines conformance gating, agent-facing workloads, strict-latency deployment rules, objective sensitivity, budget stability, and reproducibility artifacts.

## Evidence now in good shape

- The archived bundle passes strict-schema validation: 72 run directories, 0 errors.
- The report regeneration matches the staged paper tables.
- Strict 200 ms p99 deployment selects FAISS CPU with bge-small across S1/S2/S3, with decision margins now explicit.
- S3 quality-first uncertainty is handled with a full matched-query audit over 5,000 S3 queries. pgvector does not show a meaningful quality advantage over FAISS-base.
- The manuscript now records archive size, B2 metadata coverage, local hardware/runtime, and HotpotQA-portable manifest/checksum pointers.

## Remaining reviewer risks

- S2 competitor reruns are expensive. The first Qdrant HNSW64 quick attempt did not finish within 42 minutes and should not be used as evidence, but a dedicated longer standard Qdrant HNSW64 row is now complete. A paired analysis over matched observations gives Qdrant minus FAISS nDCG@10 = -0.002780 with 95% CI [-0.007268, 0.000574] and equal freshness over 500 events, supporting the S2 strict-decision story as a competitor sanity check. A same-orchestration deadline mini-bundle also completed FAISS CPU and Qdrant with two repeats per engine, zero errors, Qdrant minus FAISS nDCG@10 = -0.000734 with 95% CI [-0.002609, 0.000484], and equal freshness over 80 matched events; because it caps timed phases, it supports repeatability rather than replacing standard latency evidence.
- The paper has one local hardware profile, so p99 and cost decisions are local deployment evidence rather than production guarantees.
- The benchmark is a retrieval-infrastructure audit, not full end-to-end agent task success.

## Highest-value next experiments

1. Full uncapped S2 mini-bundle rerun.
   - Goal: rerun FAISS CPU and Qdrant under the streaming-memory workload with completed result files and repeated rows in the same orchestration, without the deadline caps.
   - Success criterion: clean comparable B2 rows with result parquet, observation logs, freshness metrics, p99, and at least two repeats per engine.

2. Second hardware profile.
   - Goal: test whether the strict-latency decision is stable across hardware.
   - Success criterion: rerun the strict winners and nearest alternatives for S1/S2/S3 on a different CPU/RAM profile.

3. Production-sized S3 stress track.
   - Goal: show how the decision changes as corpus size and evidence-set difficulty increase.
   - Success criterion: one new stress table with quality, p99, and task cost for FAISS CPU and at least one service engine.

## Writing priorities

- Keep the title and abstract decision-centered.
- Avoid universal engine superiority language.
- Keep objective sensitivity as a central result.
- Make incomplete or slow experiments visible in notes, not in the main result table.
