# MaxionBench

MaxionBench is a reproducible decision harness for LLM serving, RAG, and agents. It compares a
self-hosted vLLM fleet routed by llm-d with a managed Gemini API, and hybrids of the two, under one SLO
model. Every headline number comes from repeated trials with confidence intervals and provenance.

The earlier vector-database benchmark lives at the `v0.1` tag and the `v0.2-cpu-infra` branch.

## v0.3 harness

| Layer | Implementation |
| --- | --- |
| Experiment harness | `maxionbench/harness/`: YAML specs, matrix planner, open-loop (Poisson or replayed Azure trace) and closed-loop load generation, 95% CIs per cell, provenance, generated JSON Schema |
| Local serving | vLLM on Apple Silicon (`vllm-metal`, GPU) and llama.cpp (Metal/CPU) with Qwen3 models |
| Control plane | llm-d endpoint picker + Envoy without Kubernetes (Docker) over vllm-metal or `llm-d-inference-sim` workers |
| Managed API | `gemini-3.5-flash-lite` under a hard spend cap shared by Python and Go (`configs/pricing/gemini.yaml`) |
| AI gateway | `gateway/` (Go): local-first routing, fixed or predicted-wait overflow to Gemini, failover, Prometheus metrics, OpenTelemetry tracing |
| Evaluation | `maxionbench/eval`, `graders`, `agents`: QA with provided context (CRAG, HotpotQA) and a calibrated Gemini judge, offline BFCL AST grader, agentic HotpotQA over an MCP search/read server |
| Observability | `deploy/observability/`: OTel collector, Jaeger, Prometheus, Grafana |
| Dashboard | `dashboard/`: static TypeScript site over saved results |

```bash
. ~/.venvs/maxionbench/bin/activate
python -m maxionbench.datasets.sources fetch                     # pinned datasets, SHA-256 verified
python -m maxionbench.harness run experiments/e2_prefix_caching.yaml
python -m maxionbench.harness compare <run_dir_a> <run_dir_b>   # CI overlap between runs
python -m maxionbench.harness schema --check                    # result contract is up to date
cd dashboard && npm ci && npm run data && npm run build          # results dashboard
```

CI (`.github/workflows/v03_ci.yml`) runs lint, Python/Go/TypeScript tests, schema and type drift
checks, and a CPU performance gate (llama.cpp + inference-sim) with no paid API calls.

Paid-API keys are read only at runtime (`GEMINI_API_KEY` or a git-ignored key file) and are never
written to logs, result bundles, git, or images; every paid request reserves its worst-case cost first.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `maxionbench/` | Harness, evaluation, graders, agents, datasets, KV-cache simulator, and runtime metadata. |
| `configs/` | API pricing configuration. |
| `experiments/` | v0.3 experiment specs (E1–E6, CI smoke runs). |
| `gateway/` | Go AI gateway (routing, overflow, spend cap, metrics, tracing). |
| `dashboard/` | TypeScript results dashboard built from saved result files. |
| `deploy/` | llm-d without Kubernetes (EPP + Envoy) and the observability stack. |
| `ci/` | Performance-gate bounds checked in CI. |
| `docs/` | Project notes. |
| `tests/` | Python tests for the harness, evaluation, datasets, and simulator. |
| `dataset/processed/hotpot_portable/` | Frozen HotpotQA-MaxionBench fixture and checksums. |
| `artifacts/`, `results/`, `release/` | Local generated outputs; ignored by default and packaged explicitly when needed. |
