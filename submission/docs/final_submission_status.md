# Final Submission Status

Status date: 2026-05-04.

## Public Bundle

- PDF upload candidate: `submission/manuscript/main.pdf`.
- Source package root: `submission/manuscript/`.
- Artifact card: `submission/docs/artifact_card.md`.
- Upload checklist: `submission/docs/upload_checklist.md`.
- Dataset/evaluation metadata: `submission/metadata/`.
- Evidence summaries: `submission/evidence/`.

## Local Verification Targets

- Clean LaTeX rebuild from `submission/manuscript/`.
- Focused repository hygiene tests.
- Full project tests when local ignored artifacts and optional services are available.
- Strict-schema validation over local archived runs when `artifacts/runs/portable/` is present.

## Known Upload Gate

The abstract and full-paper upload are performed through the conference submission system, not by local repository scripts. Save the final submission confirmation and artifact hashes after upload.

## Git State Note

Public NeurIPS-facing documents intended to be visible on GitHub are mirrored under `submission/`.
