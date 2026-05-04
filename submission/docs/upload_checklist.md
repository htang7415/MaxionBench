# Upload Checklist

## Local Bundle

- `submission/manuscript/main.pdf` is the PDF upload candidate.
- `submission/title.txt` is the OpenReview title field.
- `submission/abstract.txt` is the OpenReview abstract field.
- `submission/supplement.zip` is the supplementary ZIP candidate.
- `submission/manuscript/` contains the source package needed to rebuild the PDF.
- `submission/metadata/` contains dataset/evaluation metadata.
- `submission/docs/artifact_card.md` records the public claim boundary.
- `submission/evidence/` contains the archive manifest, conformance matrix, and targeted experiment summaries used by the paper.

## Before Upload

- Rebuild `submission/manuscript/main.pdf` from a clean LaTeX state.
- Rebuild `submission/supplement.zip` after any supporting document, metadata, evidence, figure, table, or source-package change.
- Confirm page accounting against the active NeurIPS Evaluations & Datasets instructions.
- Confirm the anonymous artifact URL in `submission/metadata/hotpotqa_portable_croissant.jsonld`.
- Fix the current anonymous artifact URL before upload if it still returns HTTP 401/404 to unauthenticated reviewers.
- Confirm final repository/data licenses, especially inherited HotpotQA terms.
- Confirm no interrupted or partial runs are cited as paper evidence.
- Confirm all public GitHub submission-facing documents live under `submission/`.

## Repository Hygiene

- Do not add paper-writing or paper-preparation scripts outside `.agents/` or `.claude/`.
- Do not commit local result trees, release tarballs, reviewer-package scratch directories, LaTeX logs, editor state, AppleDouble sidecars, or cache directories.
- Do not commit host paths, user names, personal GitHub URLs, or exact machine-identifying details in public text files.

## Story Guardrails

- Say "local strict-latency decision" or "max-row p99 policy", not "best engine".
- Say "objective sensitivity", not "pgvector quality win".
- Say "paired checks rule out a hidden Qdrant quality or post-insert retrievability win", not "Qdrant is worse".
- Say "bounded repeatability evidence", not "bounded run replaces standard latency evidence".
