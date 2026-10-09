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
| AI gateway | `gateway/` (Go): local-first routing, fixed or predicted-wait overflow to Gemini, failover, cache-aware context management for agent sessions (`context:`; matches the Python policies), Prometheus metrics, OpenTelemetry tracing |
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

**Accuracy and API cost (Gemini 3.5 Flash-Lite agent on BrowseComp-Plus, 50 tasks × 7 policies, each task
searching ~800 web pages; difference vs `full` on the same tasks, 95% CI).** Accuracy is graded by the
calibrated Gemini judge and, strictly, by string match against the gold answer (they agree on 93% of
answers); p is an exact McNemar test vs `full`, Holm-adjusted over the six policies.

| Policy | Judge | Δ judge | p (Holm) | Strict | Δ strict | p (Holm) | Cost / task | Δ cost | Cost / correct | Cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 46% | – | – | 38% | – | – | $0.032 | – | $0.069 | 64% |
| `truncate` | 40% | −6 pts [−20, +8] | 1.0 | 36% | −2 | 1.0 | $0.013 | **−58%** [−93%, −22%] | **$0.033** | 68% |
| `window` | 58% | +12 [−1, +25] | 0.73 | 54% | +16 | 0.13 | $0.031 | −2% | $0.053 | 12% |
| `mask` | 52% | +6 [−6, +18] | 1.0 | 48% | +10 | 0.58 | $0.024 | −25% | $0.046 | 0% |
| `summarize` | **66%** | **+20 [+7, +33]** | **0.04** | 54% | +16 | 0.13 | $0.039 | +24% | $0.059 | 17% |
| `window+cache` | 56% | +10 [−3, +23] | 0.91 | 48% | +10 | 0.58 | $0.022 | −29% | **$0.040** | 53% |
| `mask+cache` | 56% | +10 [−4, +24] | 0.91 | 50% | +12 | 0.58 | $0.026 | −17% | $0.047 | 52% |

Five of six trimming policies were at least as accurate as `full` under both gradings, but at 50 tasks only
`summarize` under the judge is significant after correction; the accuracy gain is a consistent direction,
not an established effect. The cost results are firm: policies that rewrite earlier messages lost Gemini's
implicit prompt cache (`mask` 0% cached), while the cache-aware variants kept half the cache and cost
17–29% less than `full`.

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
python -m maxionbench.eval.context_regrade artifacts/context_eval/<run>       # strict grading, McNemar + Holm
python -m maxionbench.kvsim experiments/k6_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k7_llmd_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k8_vllm_metal_copilot_context_policies.yaml --out artifacts/kvsim
```

BrowseComp-Plus text must not be published: result files hold task ids and numbers only, and model
answers stay in a local file.

## v0.5: cache-aware context management in the gateway

v0.4 measured context policies as a client library. v0.5 puts the policy in the serving path: the Go
gateway (`gateway/internal/ctxmgr`, `context:` in its config) keeps, per agent session (`prompt_cache_key`,
else the system prompt and task), the view it last sent upstream and only appends to it, so the engine's
prefix cache keeps hitting. It trims (masks old tool results, or keeps the last N exchanges) only when the
view passes a token budget, or when the history it sees is new or rewritten and the prefix is cold anyway.
After a trim the view must grow by `min_growth` tokens before the next one. Without that, long sessions
whose trimmed view was still over budget were trimmed again on almost every call, and each trim missed
the cache. The Go code produces the same views as `agents/context.py` on a fixture generated from
Python, and the gateway reports each decision in a response header, Prometheus counters and trace
attributes.

**K9: Copilot traffic through the gateway onto a real engine.** Copilot coding-agent sessions (a Saturday
and a Tuesday, the Tuesday with ~5x the daily traffic) are replayed as full chat histories in real trace
time; message text is deterministic filler at 1/16 of each message's tokens. Requests go through the
gateway to one vllm-metal replica of Qwen3-8B (MLX 4-bit, Apple M4 GPU, 24.7k-token KV cache, ~180 tok/s cold
prefill), 6 sessions at a time, budget 100k full-scale tokens. Two seeded session draws per day; each
arm replays the same schedule as `off`.

| Arm | Sat, draw 1 | Sat, draw 2 | Tue, draw 1 | Tue, draw 2 |
| --- | --- | --- | --- | --- |
| `off` (full history), recomputed tokens/request | 290 | 197 | 301 | 303 |
| `window+cache` (8 exchanges) | **−14%** | **−30%** | **−39%** | **−27%** |
| `mask+cache` (4 exchanges, `min_growth` 25k) | −10% | +37% | −31% | −19% |
| `mask+cache` + trim after 120 s idle | −9% | +36% | −26% | −20% |
| `mask+cache` without `min_growth` | +2% | +84% | −25% | −12% |

`window+cache` cut prefill recompute in every pair, by 14–39%. Most of the gain comes from trimming
when a session first appears or the agent rewrites its own history (8–11% of requests), when the prefix is
not cached anyway. Every later call then appends to a smaller view. The budget itself fired on under 2% of
requests. On the heavier Saturday draw, full histories drove the 90th-percentile prompt to 14k tokens of a
16k context, and the server fell behind (median TTFT 80 s, 17% of requests under 5 s). `window+cache`
kept it under capacity (median TTFT 1.0 s, 92% under 5 s). Masking keeps every assistant message, so on
long sessions its views stay large and it recomputed more than `off` on that draw, though its TTFT still
fell from 80 s to 15 s. The idle trigger rarely fired (≤3.5% of requests) and added nothing over
`mask+cache`. Recomputed tokens repeat exactly across reruns. TTFT depends on other load on the host:
one trial whose engine ran at half its usual prefill speed was rerun and is reported from the rerun.

```bash
python -m maxionbench.kvsim.gateway_replay experiments/k9_gateway_context_8b.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.gateway_replay experiments/k9b_gateway_mask_growth_8b.yaml --out artifacts/kvsim
```

**C2: accuracy and API cost with the gateway in front of Gemini.** The same Flash-Lite agent as C1 sends
its full history to the gateway (Gemini only, `window+cache`: last 2 exchanges, budget 48k tokens,
`min_growth` 16k), which trims it before Gemini. C2a ran the C1 tasks and is paired with C1's runs; C2b
ran `full` and the gateway on 50 new tasks. 100 paired tasks in all:

| 100 tasks | Judge | Strict | Billed cost / task | Prompt tokens / task | Cost / correct | Cached |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 44% | 38% | $0.035 | 287k | $0.081 | 66% |
| gateway `window+cache` | 52% | 48% | −12% [−36%, +13%] | **−50%** [−80%, −20%] | $0.060 | 32% |
| difference (wins/losses, p, Holm p) | +8 (17/9, p 0.17, 0.51) | +10 (17/7, p 0.064, 0.19) | | | | |

Trimming in the gateway halved the prompt tokens Gemini billed, and accuracy did not drop. Accuracy was
higher on both task sets, but not significant at 100 tasks. Billed cost fell less than tokens: each trim
loses Gemini's implicit cache, and the cached share fell from 66% to 32%. On the C1 tasks the gateway
matched the in-agent `window+cache` policy (56% vs 56% judge, 7 wins and 7 losses). Running the policy in
the gateway costs no accuracy. Gemini rejects `prompt_cache_key` (HTTP 400), so the gateway uses it as the
session key and strips it before forwarding.

```bash
python -m maxionbench.eval.context_eval experiments/c2a_gateway_context.yaml   # Gemini; paid, resumable
python -m maxionbench.eval.context_eval experiments/c2b_gateway_context.yaml
python -m maxionbench.eval.gateway_accuracy <c1 run> <c2a run> <c2b run> --out gateway_accuracy.json
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `maxionbench/` | Harness, evaluation, graders, agents, datasets, KV-cache simulator, and runtime metadata. |
| `configs/` | API pricing configuration. |
| `experiments/` | Experiment specs: v0.3 E1–E6, v0.4 C1 (context policies on Gemini) and K1–K8 (KV cache and serving replays), v0.5 K9 (gateway context management on vllm-metal) and C2 (gateway on Gemini), CI smoke runs. |
| `gateway/` | Go AI gateway (routing, overflow, spend cap, metrics, tracing). |
| `dashboard/` | TypeScript results dashboard built from saved result files. |
| `deploy/` | llm-d without Kubernetes (EPP + Envoy) and the observability stack. |
| `ci/` | Performance-gate bounds checked in CI. |
| `docs/` | Project notes. |
| `tests/` | Python tests for the harness, evaluation, datasets, and simulator. |
| `dataset/processed/hotpot_portable/` | Frozen HotpotQA-MaxionBench fixture and checksums. |
| `artifacts/`, `results/`, `release/` | Local generated outputs; ignored by default and packaged explicitly when needed. |
