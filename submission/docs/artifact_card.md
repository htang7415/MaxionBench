# MaxionBench Artifact Card

## Intended Track

NeurIPS Evaluations & Datasets.

MaxionBench is a benchmark/evaluation package for decision-audit claims about agentic retrieval infrastructure. It is not a new retrieval model and it is not an engine leaderboard.

## Supported Claims

- Conformance-gated evaluation changes which engine rows are reportable.
- Strict-latency deployment decisions should report quality, post-insert top-10 retrievability, max-row p99 latency, normalized context-cost proxy, budget level, and objective together.
- Under the archived single-node B2 evidence and a 200 ms max-row p99 rule, FAISS CPU with `BAAI/bge-small-en-v1.5` is the policy-selected strict-latency configuration for S1, S2, and S3.
- Objective choice changes deployment conclusions, especially for S3.
- The S3 pgvector quality-first row should be interpreted as objective sensitivity because the matched-query audit is indistinguishable from FAISS.
- S2 paired checks do not show a hidden Qdrant quality or post-insert retrievability win.

## Unsupported Claims

- Universal engine superiority.
- Production latency across hardware, clusters, or managed services.
- Full end-to-end agent task success.
- Managed-vector-database deployment claims.
- Second-machine, distributed, GPU, or managed-service robustness claims.
- High-probe or high-ef service-engine sweep claims beyond the targeted checks in `submission/evidence/experiments/`.

## Reviewer Entry Points

- Manuscript PDF: `submission/manuscript/main.pdf`
- Main paper source: `submission/manuscript/main.tex`
- Evidence map: `submission/manuscript/tables/evidence_strength.tex`
- Archive manifest: `submission/evidence/archive/archive_manifest.json`
- Generated tables: `submission/tables/`
- Generated figures: `submission/figures/`
- Croissant/evaluation metadata: `submission/metadata/`
- Targeted experiment summaries: `submission/evidence/experiments/`

## Reproducibility Checks

Source compile check:

```bash
cd submission/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c
```

Project tests:

```bash
python -m pytest -q
```

Archived run validation, when local ignored artifacts are present:

```bash
python -m maxionbench.cli validate --input artifacts/runs/portable --strict-schema --json
```

Expected archived validation result for the cited bundle: 72 run directories checked, 0 errors.

## Data and License Notes

HotpotQA-MaxionBench is a bounded preprocessing of HotpotQA dev distractor for retrieval/evidence-coverage evaluation. HotpotQA is distributed under CC BY-SA 4.0 according to the official HotpotQA dataset card. The submission metadata records checksums and preprocessing metadata so reviewers can verify the exact corpus.

Before final upload, confirm that the anonymous reviewer-artifact URL in the Croissant metadata resolves to the uploaded artifact.

## Compute Scope

The reported evidence is single-node, local, CPU-only benchmark evidence. Host identifiers, OS patch level, user names, and machine-local paths are intentionally omitted from the public repository copy. The paper should therefore be read as local deployment evidence under the recorded configs, not as a universal infrastructure ranking.

## AI/Agent Use Disclosure

Code assistance and manuscript editing assistance were used during preparation. The authors remain responsible for all text, figures, references, and experimental claims. No citation should be accepted without manual verification.
