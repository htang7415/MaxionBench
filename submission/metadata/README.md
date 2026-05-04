# Metadata

This directory contains reviewer-facing metadata for the NeurIPS Evaluations & Datasets submission.

- `hotpotqa_portable_croissant.jsonld`: Croissant-style machine-readable metadata for the HotpotQA-MaxionBench evaluation corpus.
- `maxionbench_evaluation_card.json`: machine-readable claim/evidence/limitation summary for the benchmark artifact.

Current validation command:

```bash
python - <<'PY'
import mlcroissant as mlc
dataset = mlc.Dataset("paper/metadata/hotpotqa_portable_croissant.jsonld")
print(dataset.metadata.name)
print([record_set.name for record_set in dataset.metadata.record_sets])
PY
```

The expected record sets are `corpus`, `queries`, and `qrels`. Before final upload, confirm that the anonymous reviewer-artifact URL in the JSON-LD resolves to the uploaded artifact.
