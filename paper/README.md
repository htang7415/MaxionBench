# NeurIPS Paper Artifacts

This folder stages the manuscript, figures, tables, and result notes for the NeurIPS draft.

Source archive: `results/20260429T033427Z`

## Layout

- `manuscript/`: NeurIPS LaTeX source and `main.pdf`.
- `tables/`: generated CSV/TeX tables from the archived report bundle.
- `figures/`: generated PDF/PNG figures and metadata.
- `results/`: supporting result metadata used by the paper.
- `experiments/`: deadline-time sanity checks and paired analyses used to de-risk the story.
- `archive/`: archive manifest for the exact reproducibility bundle.
- `artifact_card.md`: reviewer-facing claim, scope, and reproducibility summary.
- `submission_readiness.md`: final upload checklist and story guardrails.
- `metadata/`: Croissant-style dataset metadata and evaluation-card metadata.
- `verify_neurips_artifacts.py`: quick local artifact-readiness check.

## Build the manuscript

```bash
cd paper/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c
```

Expected output: `paper/manuscript/main.pdf`.

## Validate the archived bundle

Quick paper artifact check:

```bash
python paper/verify_neurips_artifacts.py --json
```

Archived run schema check:

```bash
python -m maxionbench.cli validate \
  --input artifacts/runs/portable \
  --strict-schema \
  --json
```

Expected result used by the paper: pass, 72 run directories checked, 0 errors.

## Key paper-facing evidence

- `tables/neurips_main_results.*`: strict-latency minimum viable deployment results with confidence intervals.
- `tables/portable_decision_table.*`: deployment decisions under strict p99, unconstrained cost, and quality-first objectives.
- `manuscript/tables/strict_decision_margins.tex`: decision-margin table for the strict-latency winners.
- `manuscript/tables/s3_paired_quality.tex`: 5,000-query matched S3 audit showing no substantive pgvector quality advantage.
- `manuscript/tables/s2_competitor_check.tex`: larger same-machine paired S2 FAISS/Qdrant competitor sanity check.
- `experiments/s2_larger_same_machine/`: B2 same-machine S2 FAISS/Qdrant rerun with two repeats per engine, 1,788 matched quality observations, and 200 matched freshness events.
- `experiments/s2_mini_bundle/`: same-orchestration bounded S2 FAISS/Qdrant mini-bundle with two repeats per engine.
- `manuscript/tables/evidence_strength.tex`: reviewer-facing claim/evidence/risk map.

## Current story boundary

The paper should be read as a decision-centered benchmark paper, not as a universal engine ranking. The supported claim is that conformance-gated, single-node benchmark evidence can guide local deployment decisions when quality, freshness, p99 latency, task cost, budget stability, and objective sensitivity are reported together.
