# .agents-Guided NeurIPS Check

## Assumptions

- Target track: NeurIPS 2026 Evaluations & Datasets.
- Goal: strengthen the benchmark artifact and paper story, not reposition the work as a new retrieval algorithm.
- Current evidence should be treated as local single-machine deployment evidence.

## Formatting Check

- `paper/manuscript/main.tex` uses `\usepackage[eandd]{neurips_2026}`.
- The abstract is one paragraph.
- The manuscript compiles with the NeurIPS 2026 style file copied from `.agents/Formatting_Instructions_For_NeurIPS_2026`.
- Current PDF has 10 total pages, including appendix/checklist material. Main body content before references is within the 9-page content limit.

## Research-Paper-Writing Skill Check

- Story clarified: decision-centered benchmark for agentic retrieval infrastructure.
- Major claims mapped to evidence in `paper/manuscript/tables/evidence_strength.tex`.
- Unsupported universal-engine-ranking claims are explicitly excluded in `paper/artifact_card.md`.
- Paragraph roles and reverse outline are recorded in `paper/manuscript/self_review.md`.
- Final adversarial review has been expanded in `paper/manuscript/self_review.md`.

## Remaining Submission Risks

- Croissant metadata now uses an anonymous reviewer-artifact URL and passes `mlcroissant` validation with corpus, query, and qrel record sets parsed.
- The reviewer-artifact URL must still resolve to the uploaded anonymous artifact before submission.
- Asset/license wording should be manually checked before public release.
- The larger same-machine S2 FAISS/Qdrant run completed with zero validation errors and is included as a sanity check.

## Current Recommendation

Freeze the paper story. Spend remaining time on artifact packaging, URL resolution, license verification, visual table/figure inspection, and final checklist consistency.
