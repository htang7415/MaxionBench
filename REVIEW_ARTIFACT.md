# MaxionBench NeurIPS Review Artifact

This branch is a reviewer-facing snapshot for the NeurIPS submission. It
contains the code, manuscript sources, generated paper figures/tables, archived
portable run outputs, the larger same-machine S2 FAISS/Qdrant check, and the
HotpotQA-portable Croissant metadata inputs needed by the artifact card.

Reviewer entry points:

- `paper/artifact_card.md`
- `paper/manuscript/main.pdf`
- `paper/metadata/hotpotqa_portable_croissant.jsonld`
- `paper/verify_neurips_artifacts.py`
- `artifacts/runs/portable/`
- `artifacts/runs/s2_larger_same_machine_b2/`
- `dataset/processed/hotpot_portable/`

Quick checks:

```bash
python paper/verify_neurips_artifacts.py --json
python -m maxionbench.cli validate --input artifacts/runs/portable --strict-schema --json
python -m maxionbench.cli validate --input artifacts/runs/s2_larger_same_machine_b2 --strict-schema --json
```

The large reusable embedding arrays are intentionally excluded from this review
snapshot because the dataset metadata covers the raw corpus/query/qrel files and
the embeddings can be regenerated from the public model identifiers.
