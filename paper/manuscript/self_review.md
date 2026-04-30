# Manuscript self-review

## Mini-outline

- Opening: modern agentic AI systems use retrieval as operational memory, so deployment evidence must cover quality, freshness, latency, and context cost rather than a single leaderboard metric.
- Benchmark design: conformance gate, three workloads, two embedding tiers, three budget levels, and objective-specific deployment decisions.
- Evidence: strict 200 ms p99 selects FAISS CPU with bge-small across S1/S2/S3, but decision margins differ by workload; the S2 standard Qdrant row, bounded mini-bundle, and larger same-machine FAISS/Qdrant run support the S2 story as scoped competitor sanity checks.
- Boundary: S1/S2 are latency tie-breaks, the S3 matched-query audit does not support a substantive pgvector quality advantage, and budget stability is mixed, so the paper reports instability instead of claiming universal stability.
- Reproducibility: archived run bundle, strict validation, behavior cards, exact commands, hardware/runtime metadata, and dataset manifest/checksum pointers.

## Paragraph roles

- Abstract: one-paragraph problem, method, main numbers, and limitation.
- Introduction paragraph 1: task/application motivation.
- Introduction paragraph 2: evaluation gap.
- Introduction paragraph 3: benchmark solution.
- Introduction paragraph 4: evidence-backed findings.
- Introduction paragraph 5: contributions.
- Benchmark section: protocol and metric definitions.
- Experiments section: artifact audit, main table, decision table, strict-decision margin table, S3 matched-query audit table, S2 competitor check, figures, and quick spot checks.
- Limitations section: scope and broader impact.

## Claim-evidence map

- Claim: the benchmark is conformance-gated. Evidence: `paper/results/conformance_matrix.csv`, `paper/tables/portable_support_table.csv`, and the artifact audit paragraph. Status: supported.
- Claim: strict 200 ms p99 selects FAISS CPU with bge-small for all three workloads. Evidence: `paper/tables/neurips_main_results.csv` and `paper/tables/portable_decision_table.csv`. Status: supported.
- Claim: strict-decision margins differ across workloads. Evidence: `paper/manuscript/tables/strict_decision_margins.tex`; S1/S2 are cost/quality ties resolved by p99, while S3 has lower cost and higher evidence coverage than the next strict-cost candidate. Status: supported.
- Claim: objective choice changes winners. Evidence: `paper/tables/portable_decision_table.csv`. Status: supported with the S3 pgvector result framed as objective sensitivity, not a substantive quality advantage.
- Claim: the S3 quality-first pgvector edge is not substantive. Evidence: the matched-query audit over all 5,000 S3 queries gives pgvector minus FAISS CPU evidence_coverage@10 = -0.0001 with paired 95% interval [-0.0003, 0.0000], and pgvector p99 is much higher in that audit. Status: supported.
- Claim: the S2 strict decision is not hiding a Qdrant quality or freshness win. Evidence: the larger same-machine two-repeat FAISS/Qdrant run gives Qdrant minus FAISS nDCG@10 = -0.002671 with 95% CI [-0.005212, -0.000575] over 1,788 matched quality observations and zero paired freshness delta over 200 matched events. The earlier standard Qdrant row and bounded mini-bundle agree on equal freshness and no positive Qdrant quality edge. Status: supported as sanity checks, not universal latency evidence.
- Claim: budget stability is workload-dependent. Evidence: `paper/tables/portable_stability.csv` and decision-table B0-to-B2 columns. Status: supported.
- Claim: commodity hardware can produce useful deployment evidence. Evidence: archived complete run bundle plus strict validation and appendix metadata for the Apple M4 Mac mini runtime; wording is intentionally limited to "useful evidence", not universal stability. Status: supported with scoped language.
- Claim: the result bundle is reproducible enough for reviewer inspection. Evidence: appendix records 72 result parquet files, 24 B2 metadata files, hardware/runtime metadata, and HotpotQA-portable manifest/checksum pointers. Status: supported.

## Adversarial review

- Contribution: Pass. The paper contributes an evaluations-and-datasets style benchmark protocol and a result bundle, not a new retrieval model.
- Writing clarity: Improved. The story is now a decision-centered benchmark for agentic retrieval infrastructure rather than an engine leaderboard.
- Experimental strength: Moderate. The strict-latency result is supported and now includes decision margins; the S3 pgvector quality-first result is explicitly demoted from a quality claim after the matched-query audit.
- Evaluation completeness: Moderate. It includes four reportable engines and three workloads, improved archive metadata, S3 matched-query auditing, and a larger S2 same-machine competitor run, but only one local hardware profile.
- Method design soundness: Pass with limitations. The conformance gate and budget ladder are defensible; broader hardware and distributed settings are future work.

## .agents five-dimension checklist

### Contribution

- What new knowledge does this paper give? Pass: deployment decisions for agentic retrieval depend on conformance, freshness, p99 latency, context cost, objective, and budget stability together.
- Is the failure case meaningful? Pass: quality-only retrieval leaderboards can produce brittle deployment decisions for systems that read, write, and pack context.
- Is the idea non-obvious beyond standard practice? Pass: the contribution is the decision-audit protocol and evidence map, not another engine ranking.
- Is the gain insightful rather than predictable? Pass: objective sensitivity and fragile quality-first margins are explicitly demonstrated.
- Is there a clear novelty type? Pass: benchmark protocol, portable workloads, result bundle, and decision-margin analysis.

### Writing clarity

- Can a reader reproduce the method? Pass: the appendix, artifact card, README, archive manifest, and verifier specify commands and expected checks.
- Are key modules detailed? Pass: conformance, workloads, metrics, budget ladder, and decision objectives are defined.
- Is motivation connected to challenge? Pass: each workload maps to an agent-facing retrieval stressor.
- Are terms consistent? Pass: the paper consistently uses strict-latency, quality-first, unconstrained-cost, budget stability, and objective sensitivity.
- Does each paragraph have one message? Mostly pass: the Introduction and Experiments sections are organized by problem, gap, protocol, evidence, and limits.

### Experimental strength

- Are improvements meaningful? Pass with caveat: the paper does not claim method improvement; it claims deployment-decision differences and fragile margins.
- Is absolute performance sufficient? Pass for benchmark evidence: the results support local decision analysis, not universal production performance.
- Are results consistent across settings? Moderate: three workloads and multiple engines are included, but only one local hardware profile.
- Are failure cases reported? Pass: S1/S2 budget instability and fragile S3 quality-first margin are explicitly reported.

### Evaluation completeness

- Are key design choices ablated? Moderate: objectives, budgets, engines, embeddings, and paired audits are covered; hardware diversity remains future work.
- Are strong baselines included? Pass for portable local engines: FAISS CPU, LanceDB in-process, pgvector, and Qdrant are reportable after conformance gating.
- Are metrics sufficient? Pass: quality, freshness, p99, task cost, errors, and budget stability are reported.
- Are datasets challenging enough? Moderate: S1/S2 use D4 portable text bundles and S3 uses HotpotQA-portable; production-sized stress remains future work.
- Are protocols documented? Pass: config files, archive manifest, behavior cards, and metadata are included.

### Method design soundness

- Is the setting realistic? Pass with limits: local single-node deployment evidence is realistic for screening, not for production clusters.
- Are there hidden defects? Pass with caveat: bounded S2 mini-bundle is clearly labeled as repeatability evidence, not latency replacement evidence.
- Is robustness shown without per-case tuning? Moderate: fixed budgets and search sweeps are used, and the larger S2 same-machine check strengthens repeatability, but a fully uncapped S2 study would still be stronger.
- Do benefits outweigh complexity? Pass: the protocol adds reporting discipline and prevents overclaiming from single metrics.
- Could reviewers argue negative net benefit? Low risk if framed as an E&D benchmark; higher risk only if read as a universal engine leaderboard, which the manuscript avoids.

## Required next edits before submission

- Verify bibliography details and asset licenses before public submission.
- Visually inspect final table/figure placement after any additional compression.
- For a stronger post-deadline version, run a fully uncapped S2 FAISS/Qdrant study if time allows. The current larger same-machine run is useful repeatability evidence, but it still caps freshness events for turnaround. Do not add the interrupted Qdrant attempt as evidence, and do not add more S3 pgvector runs unless a reviewer specifically asks for replication of the matched-query audit.
