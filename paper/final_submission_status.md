# Final Submission Status

## Ready locally

- Manuscript PDF: `paper/manuscript/main.pdf`
- Clean LaTeX rebuild: passes and produces a 10-page PDF with no undefined references, undefined citations, or overfull boxes in the final log scan.
- Paper verifier: `python paper/verify_neurips_artifacts.py --json` passes with 13 checked files and 0 warnings.
- Full tests: `python -m pytest -q` passes with 182 tests and 1 third-party deprecation warning.
- Croissant metadata: `mlcroissant` loads `paper/metadata/hotpotqa_portable_croissant.jsonld` and parses `corpus`, `queries`, and `qrels`.
- Larger S2 same-machine run: strict-schema validation passes for FAISS CPU and Qdrant, with 0 errors.

## Hard blocker before upload

- The artifact URL in `paper/metadata/hotpotqa_portable_croissant.jsonld` currently redirects but returns HTTP 401. Upload the anonymous artifact, then update or unlock the URL so reviewers can access it.

## Do not change now unless necessary

- Do not add more experiments before submission.
- Do not replace the archived main tables unless the full archived bundle is regenerated consistently.
- Do not cite interrupted or partial runs as evidence.
