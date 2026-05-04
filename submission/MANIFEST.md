# Submission Manifest

This manifest records the NeurIPS-facing documents intentionally mirrored under `submission/`.

## Required Upload Documents

| File or directory | Role |
|---|---|
| `manuscript/main.pdf` | Main PDF upload candidate. |
| `manuscript/main.tex` | LaTeX entry point. |
| `manuscript/checklist.tex` | NeurIPS checklist included by `main.tex`. |
| `manuscript/references.bib` | Bibliography used by the manuscript. |
| `manuscript/neurips_2026.sty` | NeurIPS style file used to compile the source package. |
| `manuscript/sections/` | Manuscript section source. |
| `manuscript/tables/` | LaTeX table includes used by the manuscript. |
| `figures/` | Figure files referenced by the manuscript. |

## Supporting Submission Documents

| File or directory | Role |
|---|---|
| `metadata/hotpotqa_portable_croissant.jsonld` | Croissant-style metadata for HotpotQA-MaxionBench. |
| `metadata/maxionbench_evaluation_card.json` | Machine-readable benchmark claim/evidence/limitation card. |
| `metadata/README.md` | Metadata validation notes. |
| `docs/artifact_card.md` | Human-readable artifact card. |
| `docs/upload_checklist.md` | Public upload checklist. |
| `docs/final_submission_status.md` | Final local status summary. |
| `docs/repo_inventory_for_neurips.md` | Inventory of paper evidence and source locations. |

## Evidence Documents

| File or directory | Role |
|---|---|
| `evidence/archive/archive_manifest.json` | Manifest for the archived run bundle used by the paper. |
| `evidence/results/conformance_matrix.csv` | Conformance/reportability matrix used by paper tables. |
| `evidence/experiments/` | Targeted paired-audit and repeatability summaries used by appendix tables. |

## Exclusions

The submission bundle intentionally excludes paper-writing scripts, paper-preparation scripts, LaTeX build logs, AppleDouble sidecars, local result trees, release tarballs, editor state, and reviewer-package scratch directories.
