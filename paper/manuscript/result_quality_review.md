# Result quality review and NeurIPS improvement plan

This review follows the local research-paper-writing guidance: every major claim should map to evidence, unsupported claims should be weakened, and experimental weaknesses should become either new experiments or explicit limitations.

## Current judgment

The current results are good enough for an internal NeurIPS-style draft, but still need careful framing for a polished NeurIPS submission. The strongest defensible contribution is the benchmark protocol: conformance gating, agentic workloads, strict-latency deployment decisions, decision margins, and budget-stability reporting. The empirical winner story should remain deliberately narrow: strict-latency choices are local deployment decisions, not broad engine rankings, and the S3 matched-query audit removes the earlier temptation to present pgvector as a substantive quality winner.

## Supported claims

- Conformance-gated reporting is supported by `paper/results/conformance_matrix.csv` and `paper/tables/portable_support_table.csv`.
- Strict 200 ms p99 deployment selects FAISS CPU with bge-small for S1, S2, and S3 in the archived bundle.
- Budget stability is mixed: B0-to-B2 top-1 agreement is 0 for S1/S2 and 1 for S3, while full-rank Spearman is low for S3.
- The report bundle is reproducible: strict validation passes, and report regeneration matches staged paper tables.
- The S3 quality-first pgvector result is fragile rather than substantive: the full matched-query audit over 5,000 S3 queries gives pgvector minus FAISS CPU evidence_coverage@10 = -0.0001 with paired 95% interval [-0.0003, 0.0000].
- Reproducibility metadata is now explicit: the appendix records archive size, B2 metadata coverage, local hardware/runtime, and HotpotQA-portable manifest/checksum pointers.

## Weak or risky claims

- Quality-first S3 choosing pgvector in the archived decision table is a table-level objective-sensitivity result, not a quality result. The stronger matched-query audit shows no meaningful pgvector advantage and much worse p99.
- S1 strict selection between FAISS-small and LanceDB-small is a cost/quality tie broken by latency. This is fine, but the paper should call it a deployment decision under a tie-break rule rather than a broad superiority claim.
- S2 has only fixed read/write pins and expensive freshness probes, so quick reruns are not cheap. The freshness result is useful, but it should be framed as a protocol demonstration plus archived result, not a complete streaming-memory benchmark suite.
- All experiments use one local hardware profile. That is acceptable for the paper's scope only if the title, abstract, and limitations stay explicitly single-node and commodity-hardware focused.

## Quick experiments completed

- Strict-schema validation: passed, 72 run directories, 0 errors.
- Report regeneration: passed; key regenerated CSVs match staged `paper/tables` copies.
- S1 FAISS-small B2 rerun: reproduced nDCG@10 = 0.505506; max p99 = 27.663 ms.
- S3 FAISS-small B2 rerun: reproduced evidence_coverage@10 = 0.851500; max p99 = 20.638 ms.
- S3 FAISS-base B2 rerun: evidence_coverage@10 = 0.871200; max p99 = 72.018 ms.
- S3 LanceDB-small B2 rerun attempt: stopped after about 14 minutes with partial observations and no completed results, so it is not usable as confirmation.
- S3 pgvector-base versus FAISS-base aggregate-row bootstrap: mean evidence_coverage@10 difference = 0.00016, 95% interval = -0.00280 to 0.00304. This is not enough to claim a meaningful pgvector quality advantage.
- S3 pgvector-base versus FAISS-base matched-query audit: all 5,000 S3 queries checked once per setting; pgvector IVF32 and IVF64 both trail FAISS HNSW32 by 0.0001 evidence_coverage@10 with paired 95% interval [-0.0003, 0.0000].
- S2 FAISS-small observation-level check: an interrupted overbroad run still produced one complete FAISS HNSW32 observation file with nDCG@10 = 0.505506, freshness_hit@5s = 0.83, p99 = 15.129 ms, and 0 errors. It is useful as a local spot check, but not as a replacement for the archived S2 table because no comparable completed competitor row was produced.
- S2 Qdrant-small HNSW64 attempt: interrupted at 42:11 with no result or observation files while still in freshness probing. This is not usable evidence and should only be cited as a reason not to force opportunistic S2 competitor reruns into the manuscript.
- S2 Qdrant-small HNSW64 dedicated standard rerun: completed with result parquet and 1,394 observation lines; nDCG@10 = 0.502726, freshness_hit@5s = 0.83, p99 = 44.534 ms, and 0 errors. Matched against the FAISS observation-level spot check, Qdrant minus FAISS nDCG@10 = -0.002780 with paired 95% CI [-0.007268, 0.000574], and freshness_hit@5s delta is exactly 0.000 over 500 matched events.

## Improvement plan

1. Reframe the manuscript around benchmark quality, not engine victory.
   - Keep the main claim: \sys provides conformance-gated, reproducible local deployment evidence.
   - Avoid language implying FAISS is universally best or pgvector is a meaningful quality winner.
   - Treat objective sensitivity as the finding: different objectives can change the decision.

2. Add a compact "decision margin" table.
   - Show strict winner, nearest strict alternative, quality delta, p99 delta, and whether the result is a tie-break.
   - This directly addresses reviewer skepticism about tiny differences.
   - It is more useful than adding more large figures.

3. Keep S3 quality-first uncertainty explicit.
   - The matched-query audit is now strong enough to remove the substantive pgvector quality claim.
   - The manuscript should say the archived pgvector row demonstrates objective sensitivity, while the direct audit shows no meaningful quality advantage over FAISS-base.

4. Run only high-value experiments before submission.
   - Additional S3 pgvector reruns are no longer the highest-value next experiment because the matched-query audit already answers the main reviewer risk.
   - Completed: one dedicated standard Qdrant HNSW64 rerun now provides a comparable S2 competitor sanity check. A full new bundle would still require rerunning both engines/repeats under the same orchestration.
   - Do not run full-matrix reruns unless preparing a new archive.

5. Improve reproducibility presentation.
   - Completed: the appendix now includes hardware/runtime metadata from `run_metadata.json`, archive/result coverage, and dataset manifest/checksum pointers.
   - Completed: quick reruns are explicitly presented as sanity checks, not replacements for the archived bundle.

6. Strengthen limitations.
   - Completed: limitations now state that confidence intervals are query/event-level, not hardware-population intervals.
   - Completed: limitations now state that p99 is observed under one machine profile and not a production tail-latency guarantee.
   - Completed: limitations state that S3 is a bounded HotpotQA proxy, not full-Wikipedia retrieval.

## Decision

Do not spend more time on broad reruns before revising the paper story. The fastest route toward NeurIPS quality is to keep the manuscript decision-centered, use the decision-margin and S3 matched-query tables as reviewer-facing safeguards, and treat slow or incomplete competitor reruns as limitations rather than forcing them into the main evidence.
