# NeurIPS Submission Bundle

This directory is the public GitHub handoff location for NeurIPS-facing submission documents.

It mirrors the current upload-facing materials from the local `paper/` staging tree while avoiding paper-preparation scripts, local build logs, editor state, reviewer-package scratch directories, and machine-identifying details.

## Layout

| Path | Contents |
|---|---|
| `manuscript/main.pdf` | Current PDF upload candidate. |
| `manuscript/` | LaTeX source package, including `main.tex`, `checklist.tex`, `references.bib`, `neurips_2026.sty`, section files, and manuscript table includes. |
| `figures/` | Figure PDFs/PNGs and metadata sidecars referenced by the manuscript. |
| `tables/` | Generated CSV/TeX table sources and table-facing figure copies. |
| `metadata/` | Croissant-style dataset metadata and evaluation-card metadata. |
| `evidence/` | Archive manifest, conformance matrix, and targeted experiment summaries used by the paper claims. |
| `docs/` | Artifact card, upload checklist, final status, and repository inventory. |

## Upload Mapping

- Main paper PDF: `manuscript/main.pdf`
- Source package root: `manuscript/`
- Dataset/evaluation metadata: `metadata/`
- Claim and reproducibility card: `docs/artifact_card.md`
- Evidence summaries for reviewer traceability: `evidence/`

## Local Checks

From the repository root:

```bash
cd submission/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c
```

From the repository root:

```bash
python -m pytest -q
python -m maxionbench.cli validate --input artifacts/runs/portable --strict-schema --json
```

The validation command expects the local ignored run artifacts to be present. Those artifacts are not committed to GitHub.
