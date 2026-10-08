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

## v0.4: context policies for agents

What should an agent send the model each step: the whole history, or a trimmed view? v0.4 measures
five context policies and cache-aware variants on three things at once: task accuracy, managed-API
cost, and self-hosted serving cost, grounded in production agent traffic.

| Policy | What the model sees |
| --- | --- |
| `full` | the whole history (baseline) |
| `truncate` | every tool result cut to its first 2k tokens |
| `window` | only the last N exchanges |
| `mask` | tool results older than the last N exchanges replaced by a placeholder |
| `summarize` | above a token trigger, older exchanges replaced by an LLM-written summary |
| `window+cache`, `mask+cache` | append-only view; the base policy is applied only when the view passes a token budget |

Policies live in `maxionbench/agents/context.py` (property-tested: task kept, tool calls paired with
results, cache-aware views only append between edits) and plug into the MCP agent loop.

**Production agents (GitHub Copilot coding-agent traces, June 1–7 2026: 301k sessions, 9.1M calls).**
Median prompt 57k tokens and 76% of calls above 30k; tool output is 48% of prompt tokens. Copilot rarely
cuts context (8.6% of sessions, at ~130k tokens), and the call after a cut is 27% cached versus 89% for
steady calls. Cache hits fall with the pause before a call: 93% under 10 s, 77% at 1–5 min, 23% at
5–60 min; the median pause between turns is 170 s.

**Accuracy and API cost (Gemini 3.5 Flash-Lite agent on BrowseComp-Plus, 49 tasks × 7 policies, each task
searching ~800 web pages; difference vs `full` on the same tasks, 95% CI).**

| Policy | Accuracy | Δ accuracy | Cost / task | Δ cost | Cost / correct | Cached |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 45% | – | $0.032 | – | $0.071 | 64% |
| `truncate` | 41% | −4 pts [−18, +10] | $0.013 | **−59%** [−94%, −23%] | **$0.032** | 68% |
| `window` | 57% | +12 [−1, +26] | $0.031 | −2% | $0.055 | 12% |
| `mask` | 51% | +6 [−6, +18] | $0.024 | −25% | $0.047 | 0% |
| `summarize` | **65%** | **+20 [+8, +33]** | $0.040 | +24% | $0.061 | 17% |
| `window+cache` | 55% | +10 [−3, +23] | $0.023 | −30% | **$0.041** | 53% |
| `mask+cache` | 55% | +10 [−4, +24] | $0.027 | −17% | $0.049 | 52% |

Less context made this agent more accurate (`summarize` +20 points), while the policies that rewrite
earlier messages lost Gemini's implicit prompt cache (`mask` 0% cached). The cache-aware variants kept
half the cache and cost 17–30% less than `full`.

**Serving cost (Copilot sessions replayed under each policy as prefix-chained KV blocks; prefill
recomputed vs `full`).** K6 is the offline KV simulator (4 replicas, 32 sessions, llm-d-style
prefix+load routing, 3 repeats); K7 is a live replay through the real llm-d EPP + Envoy over 4
inference-sim workers at the tight capacity; K8 replays through llm-d onto a real engine (one
vllm-metal replica of Qwen3-0.6B on the Apple GPU, prompts scaled to 1/32, same KV pressure per session).

| Policy | Prompt size | K6, 512k KV/replica | K6, 1M | K6, 64M | K7 live, 512k | K7 TTFT p99 | K8 vllm-metal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 68k | – | – | – | – | 1.49 s | – |
| `truncate` | 56k | −37% | −28% | −20% | −60% | 0.77 s | −46% |
| `window` | 21k | +20% | +134% | **+162%** | +34% | 0.84 s | +71% |
| `mask` | 36k | −14% | +45% | +64% | −16% | 0.70 s | +13% |
| `summarize` | 36k | −15% | +12% | +11% | −35% | 1.11 s | −6% |
| `window+cache` | 32k | **−53%** | −10% | +5% | −58% | 0.58 s | −36% |
| `mask+cache` | 44k | −46% | −14% | −2% | −58% | 0.61 s | −41% |

With ample KV memory, policies that rewrite the prompt make the server recompute more despite sending
50–70% fewer tokens; under KV pressure, smaller contexts avoid evictions. Cache-aware variants are the
only policies that win under pressure without losing much otherwise. Live hit ratios match the
simulator within ~3 points; on the real engine, `mask+cache` met the 1 s TTFT target for 98% of requests vs 90% for `full`.

```bash
python -m maxionbench.datasets.sources fetch --group copilot copilot_traces browsecomp_plus
python -m maxionbench.eval.copilot_characterize --jobs 3                     # production characterization
python -m maxionbench.eval.context_eval experiments/c1_context_policies.yaml  # Gemini; paid, resumable
python -m maxionbench.kvsim experiments/k6_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k7_llmd_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k8_vllm_metal_copilot_context_policies.yaml --out artifacts/kvsim
```

BrowseComp-Plus text must not be published: result files hold task ids and numbers only, and model
answers stay in a local file.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `maxionbench/` | Harness, evaluation, graders, agents, datasets, KV-cache simulator, and runtime metadata. |
| `configs/` | API pricing configuration. |
| `experiments/` | Experiment specs: v0.3 E1–E6, v0.4 C1 (context policies on Gemini) and K1–K8 (KV cache and serving replays), CI smoke runs. |
| `gateway/` | Go AI gateway (routing, overflow, spend cap, metrics, tracing). |
| `dashboard/` | TypeScript results dashboard built from saved result files. |
| `deploy/` | llm-d without Kubernetes (EPP + Envoy) and the observability stack. |
| `ci/` | Performance-gate bounds checked in CI. |
| `docs/` | Project notes. |
| `tests/` | Python tests for the harness, evaluation, datasets, and simulator. |
| `dataset/processed/hotpot_portable/` | Frozen HotpotQA-MaxionBench fixture and checksums. |
| `artifacts/`, `results/`, `release/` | Local generated outputs; ignored by default and packaged explicitly when needed. |
