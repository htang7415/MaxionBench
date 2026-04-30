# Repository Inventory for NeurIPS

Task source: `improvements.md` Task 1.

Scope: inventory only. This file records where the paper, generated evidence, run artifacts, metadata, and generation code live so later NeurIPS edits can stay tied to exact sources.

## Paper source

| Path | Purpose |
|---|---|
| `paper/manuscript/main.tex` | Main NeurIPS LaTeX entry point; includes the abstract, section files, bibliography, appendix, and checklist. |
| `paper/manuscript/sections/01_introduction.tex` | Introduction and contribution framing. |
| `paper/manuscript/sections/02_benchmark.tex` | Benchmark design, systems under test, workloads, metrics, and budget ladder. |
| `paper/manuscript/sections/03_experiments.tex` | Main experimental narrative and table/figure includes. |
| `paper/manuscript/sections/04_limitations.tex` | Limitations and broader-impact discussion. |
| `paper/manuscript/sections/05_conclusion.tex` | Conclusion. |
| `paper/manuscript/sections/appendix.tex` | Reproducibility details and appendix claim-evidence map. |
| `paper/manuscript/references.bib` | Bibliography used by the manuscript. |
| `paper/manuscript/checklist.tex` | NeurIPS checklist included after the appendix. |
| `paper/manuscript/main.pdf` | Current built PDF artifact. |

## Generated paper tables

| Path | Purpose |
|---|---|
| `paper/tables/neurips_main_results.csv` | CSV source for Table 1 strict-latency main results. |
| `paper/tables/neurips_main_results.tex` | Generated LaTeX for Table 1; mirrored into `paper/manuscript/tables/neurips_main_results.tex`. |
| `paper/tables/portable_decision_table.csv` | CSV source for Table 2 objective-sensitivity decisions. |
| `paper/tables/portable_decision_table.tex` | Generated LaTeX for Table 2; mirrored into `paper/manuscript/tables/portable_decision_table.tex`. |
| `paper/manuscript/tables/s3_paired_quality.tex` | Table 3 staged LaTeX for the matched S3 pgvector-vs-FAISS quality audit. |
| `paper/manuscript/tables/strict_decision_margins.tex` | Table 4 staged LaTeX for strict-decision margin interpretation. |
| `paper/manuscript/tables/s2_competitor_check.tex` | Table 5 staged LaTeX for the larger same-machine S2 FAISS-vs-Qdrant check. |
| `paper/manuscript/tables/evidence_strength.tex` | Reviewer-facing claim/evidence/risk map currently included in the appendix. |

## Table and figure generation code

| Path | Purpose |
|---|---|
| `maxionbench/cli.py` | `report --mode portable-agentic` CLI entry point; dispatches to the portable report bundle generator. |
| `maxionbench/reports/portable_exports.py` | Creates the portable report bundle, including Tables 1-2 CSV/LaTeX and Figures 1-2 image files. |
| `maxionbench/reports/portable_exports.py::_neurips_main_results_table` | Builds Table 1 data from archived run results, query observations, S2 event observations, and stability summaries. |
| `maxionbench/reports/portable_exports.py::_neurips_main_results_latex` | Renders Table 1 LaTeX. |
| `maxionbench/reports/portable_exports.py::_portable_decision_table` | Builds Table 2 objective-sensitivity rows. |
| `maxionbench/reports/portable_exports.py::_portable_decision_table_latex` | Renders Table 2 LaTeX. |
| `maxionbench/reports/portable_exports.py::_export_portable_figures` | Writes the figure files used by Figure 1 and Figure 2. |
| `maxionbench/reports/portable_exports.py::_plot_task_cost_by_budget` | Produces `portable_task_cost_by_budget.*`, the left panel of Figure 1. |
| `maxionbench/reports/portable_exports.py::_plot_budget_stability` | Produces `portable_budget_stability.*`, the right panel of Figure 1. |
| `maxionbench/reports/portable_exports.py::_plot_s2_post_insert_retrievability` | Produces `portable_s2_post_insert_retrievability.*`, the left panel of Figure 2. |
| `maxionbench/reports/portable_exports.py::_plot_mvd_sensitivity` | Produces `portable_mvd_sensitivity.*`, the right panel of Figure 2. |
| `maxionbench/reports/plots.py` | Shared report-result loading and figure style constants used by portable report generation. |
| `paper/experiments/s2_larger_same_machine/summarize.py` | Summarizes the S2 FAISS-vs-Qdrant rerun that feeds Table 5. |
| `paper/experiments/s3_paired_quality/summary.json` | Staged summary source for Table 3 matched S3 quality-audit values. |
| `paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json` | Staged summary source for Table 5 larger same-machine S2 values. |

## Generated figure artifacts

| Path | Purpose |
|---|---|
| `paper/figures/portable_task_cost_by_budget.pdf` | Figure 1 left panel in the manuscript. |
| `paper/figures/portable_budget_stability.pdf` | Figure 1 right panel in the manuscript. |
| `paper/figures/portable_s2_post_insert_retrievability.pdf` | Figure 2 left panel in the manuscript. |
| `paper/figures/portable_mvd_sensitivity.pdf` | Figure 2 right panel in the manuscript. |
| `paper/figures/*.meta.json` | Metadata sidecars for paper-facing figures. |
| `artifacts/figures/final/` | Current report-generation output directory copied into the paper staging area and archived bundle. |

## Run artifacts and archive

| Path | Purpose |
|---|---|
| `artifacts/runs/portable/` | Live portable B0/B1/B2 run tree consumed by report generation and validation. |
| `artifacts/runs/portable/b0/` | Budget B0 run artifacts. |
| `artifacts/runs/portable/b1/` | Budget B1 run artifacts. |
| `artifacts/runs/portable/b2/` | Budget B2 run artifacts. |
| `artifacts/runs/portable/**/results.parquet` | Per-run measured result rows. |
| `artifacts/runs/portable/**/run_metadata.json` | Per-run metadata including model, hardware/runtime, budget, `c_llm_in`, and RHU references. |
| `artifacts/runs/portable/**/config_resolved.yaml` | Fully resolved per-run configuration, including cost coefficients, embedding model, workload caps, and search sweep. |
| `artifacts/runs/portable/**/run_status.json` | Per-run success/failure status used by report loading. |
| `results/20260429T033427Z` | Archived portable run directory used by the manuscript. |
| `results/20260429T033427Z.tar.gz` | Compressed archive for the manuscript's primary result bundle. |
| `results/20260429T033427Z/archive_manifest.json` | Archive manifest listing copied docs, runs, figures, HotpotQA-portable files, and conformance artifacts. |
| `paper/archive/archive_manifest.json` | Paper-staged copy of the archive manifest. |

## Conformance artifacts

| Path | Purpose |
|---|---|
| `artifacts/conformance/conformance_matrix.csv` | Live conformance status matrix used by reportability filtering. |
| `artifacts/conformance/conformance_matrix.json` | JSON conformance matrix. |
| `artifacts/conformance/conformance_matrix.provenance.json` | Provenance for conformance execution. |
| `artifacts/conformance/adapter_logs/` | Per-adapter stdout/stderr logs from conformance runs. |
| `results/20260429T033427Z/conformance/` | Archived conformance artifacts bundled with the primary result archive. |
| `paper/results/conformance_matrix.csv` | Paper-staged conformance matrix. |
| `docs/behavior/` | Behavior cards required for an engine to be reportable. |

## Dataset manifests and checksums

| Path | Purpose |
|---|---|
| `dataset/manifest.json` | Top-level dataset manifest. |
| `dataset/D4/beir/scifact/` | Local BEIR-format SciFact source bundle for S1/S2. |
| `dataset/D4/beir/fiqa/` | Local BEIR-format FiQA source bundle for S1/S2. |
| `dataset/D4/crag/crag_task_1_and_2_dev_v4.first_500.jsonl` | CRAG-500 event-stream source for S2. |
| `dataset/D4/hotpotqa/hotpot_dev_distractor_v1.json` | Raw HotpotQA dev distractor source used to build HOTPOTQA-PORTABLE. |
| `dataset/processed/hotpot_portable/manifest.json` | HOTPOTQA-PORTABLE manifest; records 66,635 documents, 7,405 questions, and 14,810 qrels. |
| `dataset/processed/hotpot_portable/checksums.json` | HOTPOTQA-PORTABLE SHA-256 checksums. |
| `dataset/processed/hotpot_portable/corpus.jsonl` | HOTPOTQA-PORTABLE corpus. |
| `dataset/processed/hotpot_portable/queries.jsonl` | HOTPOTQA-PORTABLE queries. |
| `dataset/processed/hotpot_portable/qrels.tsv` | HOTPOTQA-PORTABLE qrels. |
| `dataset/processed/hotpot_portable/meta.json` | HOTPOTQA-PORTABLE preprocessing metadata. |
| `results/20260429T033427Z/hotpot_portable/manifest.json` | Archived HOTPOTQA-PORTABLE manifest. |
| `results/20260429T033427Z/hotpot_portable/checksums.json` | Archived HOTPOTQA-PORTABLE checksums. |

## Cost configuration and computation

| Path | Purpose |
|---|---|
| `maxionbench/orchestration/config_schema.py` | Defines `RunConfig`, including `c_llm_in`, RHU reference constants, RHU weights, and workload caps. |
| `maxionbench/orchestration/runner.py::_portable_payload` | Computes `retrieval_cost_est`, `embedding_cost_est`, `llm_context_cost_est`, and `task_cost_est`. |
| `maxionbench/metrics/cost_rhu.py` | RHU normalization helpers and default RHU references/weights. |
| `artifacts/runs/portable/**/config_resolved.yaml` | Per-run resolved cost coefficients and RHU reference values. |
| `artifacts/runs/portable/**/run_metadata.json` | Per-run recorded `c_llm_in`, resource profile, RHU references, and RHU weights. |
| `paper/tables/portable_summary.csv` | Paper-staged per-run summary including task-cost fields used by current reports. |

## Croissant and reviewer metadata

| Path | Purpose |
|---|---|
| `paper/metadata/hotpotqa_portable_croissant.jsonld` | Croissant-style metadata for HOTPOTQA-PORTABLE. |
| `paper/metadata/maxionbench_evaluation_card.json` | Evaluation-card metadata for the benchmark artifact. |
| `paper/metadata/README.md` | Notes for reviewer-facing metadata files. |
| `paper/artifact_card.md` | Reviewer-facing artifact card with claim boundary and reproducibility summary. |
| `paper/verify_neurips_artifacts.py` | Local artifact-readiness verifier used before submission. |

## Reproduction and validation entry points

| Path | Purpose |
|---|---|
| `README.md` | Top-level project overview and primary workflow commands. |
| `command.md` | Command notes copied into the primary archive. |
| `project.md` | Portable-agentic benchmark specification copied into the primary archive. |
| `document.md` | Technical document and paper blueprint copied into the primary archive. |
| `maxionbench/tools/portable_workflow.py` | End-to-end setup, data, and finalization workflow. |
| `maxionbench/tools/submit_portable.py` | Budget-run orchestration for B0/B1/B2. |
| `maxionbench/tools/validate_outputs.py` | Strict-schema validator for run artifacts. |
| `tests/test_portable_reports.py` | Tests for report table/figure generation and reportability filtering. |
| `tests/test_result_schema.py` | Tests for result schema expectations. |
| `tests/test_verify_dataset_manifests.py` | Tests for dataset manifest verification. |
