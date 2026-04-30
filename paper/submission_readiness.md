# NeurIPS Submission Readiness

## Current Submission Position

MaxionBench should be submitted as an Evaluations & Datasets paper. The strongest story is a decision-audit benchmark for agentic retrieval infrastructure, not an engine leaderboard.

## Ready

- NeurIPS LaTeX manuscript under `paper/manuscript/`.
- Main paper PDF under `paper/manuscript/main.pdf`.
- Archived run bundle recorded as `results/20260429T033427Z`.
- Strict-schema validation for the archived portable runs: 72 run directories, 0 errors.
- Claim-evidence map in `paper/manuscript/tables/evidence_strength.tex`.
- S3 matched-query audit over 5,000 queries.
- S2 standard Qdrant paired check.
- S2 same-orchestration bounded mini-bundle with two repeats per engine.
- S2 larger same-machine FAISS/Qdrant run with two repeats per engine, strict-schema validation, and paired summary.
- Artifact card under `paper/artifact_card.md`.
- Croissant-style metadata under `paper/metadata/`, validated with `mlcroissant`.
- Local artifact verifier under `paper/verify_neurips_artifacts.py`.

## Must Do Before Upload

- Confirm that the anonymous reviewer-artifact URL resolves to the uploaded artifact. Current check: the URL redirects but returns HTTP 401, so it is not reviewer-accessible yet.
- Run `python paper/verify_neurips_artifacts.py --json` and require `pass: true`.
- Run `python -m pytest -q`.
- Compile `paper/manuscript/main.pdf` from a clean LaTeX state.
- Confirm final repository/data licenses, especially inherited HotpotQA terms.
- Confirm that no interrupted or partial runs are cited as evidence.

## Latest Local Checks

- `python paper/verify_neurips_artifacts.py --json`: pass, 13 files checked, 0 warnings.
- `mlcroissant` validation: pass; parsed record sets are `corpus`, `queries`, and `qrels`.
- `python -m maxionbench.cli validate --input artifacts/runs/neurips_rerun/s2_larger_same_machine_b2 --strict-schema --json`: pass, 2 run directories checked, 0 errors.
- `python -m pytest -q`: 182 passed, 1 warning.
- Clean LaTeX rebuild (`latexmk -C main.tex`, then `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`): pass, 10-page PDF; final log scan has no undefined references, undefined citations, or overfull boxes.
- macOS sidecar scan: pass after deleting generated `._*` files.

## Optional If Time Allows

- Optionally run an uncapped S2 study after submission-deadline work if more time is available.
- Do not replace the archived main tables unless the full bundle is regenerated consistently.

## Story Guardrails

- Say "local strict-latency decision", not "best engine".
- Say "objective sensitivity", not "pgvector quality win".
- Say "paired checks rule out a hidden Qdrant quality/freshness win", not "Qdrant is worse".
- Say "bounded S2 mini-bundle supports repeatability", not "bounded run replaces standard latency evidence".
