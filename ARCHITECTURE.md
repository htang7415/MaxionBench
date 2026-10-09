# Architecture

MaxionBench is a Python package plus a Go gateway and a TypeScript dashboard. The earlier
vector-database benchmark lives at the `v0.1` and `v0.2` tags.

## v0.3 serving harness

```text
experiments/*.yaml
 └─ harness.spec ─► harness.planner ─► harness.runner
                                         ├─ harness.targets    (system under test: lifecycle, endpoints, picker)
                                         ├─ harness.workloads  (request streams + load-generation settings)
                                         ├─ rag.loadgen        (open loop: Poisson or explicit arrivals; closed loop)
                                         ├─ harness.budget     (reserve/commit ledger for paid APIs)
                                         └─ harness.provenance (git state, host, tool versions, key redaction)
                                       ─► results.json (+ requests.jsonl, spec.yaml, logs/)
                                       ─► harness.compare / harness.perf_gate / harness.dashboard_export
```

- **Targets** own lifecycle and expose OpenAI-compatible endpoints; they never define workload or SLO
  policy. Kinds: `vllm_metal`, `llamacpp_replicas` (Metal or CPU), `llmd` (EPP + Envoy without
  Kubernetes over vllm-metal or inference-sim workers), `sim_replicas` (inference-sim with client-side
  routing), `static_endpoints`, `gemini`, and `ai_gateway` (wraps any local target).
- **Workloads** produce request streams and never know which engine serves them: `rag_sessions`
  (multi-turn HotpotQA sessions sharing a context), `synthetic_chat`, and `trace_replay` (Azure LLM
  inference trace 2024 arrivals and token lengths, scaled in rate and size).
- **Latency** is measured from the scheduled arrival time, so queueing delay is never hidden; rejected
  and failed requests stay in the denominator.
- **Repeats and CIs**: each cell's metrics are means over repeats with a 95% Student-t interval.
  Trials record whether the host was quiet (load and foreign inference processes) before they ran.
- `harness/result.schema.json` is generated from the result dataclasses and is the dashboard contract;
  `python -m maxionbench.harness schema --check` fails when it is stale.
- **Spend**: every paid request reserves its worst-case cost in a JSONL ledger outside the repo before
  it is sent, then commits provider-reported usage. Runners that share the cap with the Go gateway at the
  same time (`eval.context_eval`) reserve per request, as the gateway does, never for a whole run. Billed output is visible plus thinking tokens
  (Gemini's OpenAI-compatible usage reports thinking only in `total_tokens`). HTTP error responses
  are not billed; timeouts are charged their estimate.
- **Keys** are loaded only at runtime via `harness.secrets`; results record key presence, not value.

## Datasets and evaluation

- `maxionbench/datasets/sources.py` downloads and derives the datasets into `dataset/v03/`, and
  `maxionbench/datasets/manifests/v03.yaml` pins every file by SHA-256, in groups that `fetch --group`
  selects: `crag` (CRAG-500), `beir` (SciFact/FiQA), `sharegpt`, `azure_trace` (Azure LLM trace 2024),
  `bfcl` (BFCL v3), `agentx` (agent KV traces), `copilot` (GitHub Copilot coding-agent traces, June 1–7
  2026), `copilot_traces` (those sessions rendered under each context policy), and `browsecomp_plus`.
  Loaders verify a file before reading it.
- `maxionbench/eval/e5.py` runs one model on QA with provided context (CRAG search snippets; HotpotQA
  gold paragraphs plus distractors), BFCL v3 single-turn tool calls, and agentic HotpotQA. Requests run
  at concurrency 1; items are split into seeded shards that act as repeats in the result schema.
- `maxionbench/eval/e6.py` compares Gemini implicit caching, explicit `cachedContents`, and the inline
  Batch API on one shared-document workload.
- Graders (`maxionbench/graders/`): BFCL AST checker; EM/F1 and CRAG's three-way score; agent task
  success; an LLM judge whose rubric is calibrated against `graders/calibration/qa_judge_v1.jsonl`.
- Agents (`maxionbench/agents/`): an MCP stdio server exposing `search`/`read` over the HotpotQA
  corpus, a BrowseComp-Plus environment (BM25 search returning whole pages), and an agent loop that
  replays the model's own assistant message each turn (Gemini 3 thought signatures must come back
  unchanged).

## Agent context policies (v0.4)

- `maxionbench/agents/context.py`: what the agent sends each step — `full`, `truncate`, `window`,
  `mask`, `summarize`, and `CacheAware` (`<policy>+cache`): an append-only view re-rendered by the base
  policy only past a token budget, with `min_growth` before the next trim. Property-tested: the task is
  kept, tool calls stay paired with their results, cache-aware views only append between edits.
- `maxionbench/eval/copilot_characterize.py` characterizes the Copilot traces (prompt sizes, tool-output
  share, context cuts, cache hits against pause length).
- `maxionbench/eval/context_eval.py` runs every policy on every BrowseComp-Plus task with a Gemini agent
  (task-major, so a budget stop leaves complete pairs; resumable per (task, policy)); cost is what
  Gemini bills, correctness comes from the calibrated judge. `context_regrade.py` adds strict grading and
  Holm-corrected exact McNemar tests. Results hold task ids and numbers only; answers stay in a local
  `answers.jsonl` (BrowseComp-Plus text must not be published).
- `maxionbench/kvsim/`: an offline KV-cache simulator over prefix-chained 64-token blocks (`sim.py`;
  retention, CPU tier, routing; AgentX traces via `traces.py`, Copilot sessions under a policy via
  `copilot.py`), and `live.py`, which replays the same sessions in wall-clock time through llm-d over
  inference-sim or vllm-metal workers.

## Gateway context management (v0.5)

- `gateway/internal/ctxmgr` (Go) applies `window+cache` or `mask+cache` in the request path. A session
  is keyed by `prompt_cache_key`, else the system prompt and task; sessions that share a key keep
  separate state, and a request continues the one whose history it extends. The view stays append-only
  until it passes `budget_tokens` (then the base policy re-renders it), or the history is new or
  rewritten; `pause_s` optionally trims after an idle gap. A Python-generated fixture keeps Go and
  `agents/context.py` identical.
- `maxionbench/kvsim/gateway_replay.py` (K9) replays Copilot sessions as full chat histories (filler
  text at a set scale) through the gateway onto a real engine and reads prefix-cache hits and TTFT from it.
- In `context_eval`, a policy with a `gateway:` block runs the agent through the gateway (`remote_only`
  to Gemini); `pair_with` and `exclude_tasks_from` pair a new arm with an earlier run's tasks or draw new
  ones. `eval/gateway_accuracy.py` combines the runs into paired accuracy, cost, and token comparisons.

## AI gateway (`gateway/`, Go)

- OpenAI-compatible proxy in front of a local fleet (vLLM replicas or an llm-d gateway) with policies
  `local_only`, `local_first` (overflow beyond an in-flight threshold), `local_first_slo` (overflow
  when in-flight × recent time per completed request exceeds `slo_ttft_s`), and `remote_only`.
  Failover retries a local connection error or 5xx on the remote before any byte reaches the client.
- Paid requests reserve worst-case cost in the same JSONL ledger the Python harness uses (exclusive
  flock), so one hard cap holds across languages; actual usage, including thinking tokens, is
  committed from the response.
- The gateway process is the only holder of the provider key; it is redacted from forwarded errors and
  never appears in metrics or traces. `X-Maxionbench-Backend` attributes each response.
- Optional context management (`context:`; see v0.5 above) rewrites agent requests before routing and
  reports each decision in a response header, Prometheus counters, and span attributes. Fields the
  provider rejects (engine-only fields and `prompt_cache_key`) are stripped before remote calls.
- Prometheus metrics (requests, route decisions, in-flight, time to first byte, spend, budget,
  predicted local wait) and OpenTelemetry tracing: W3C `traceparent` is always propagated to the
  fleet and the remote; spans are exported over OTLP only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

## Observability and dashboard

- `deploy/observability/`: OpenTelemetry collector → Jaeger, Prometheus scraping the gateway, the
  llm-d EPP, vLLM, and llama.cpp, and Grafana with a provisioned serving dashboard. Every port binds
  to 127.0.0.1.
- `dashboard/`: a static Vite + React + Recharts site with pages for engines, caching, llm-d
  scheduling, hybrid serving, quality, agent context policies (v0.4), gateway context management (v0.5),
  and run provenance. `npm run data` exports the latest complete run of each published experiment
  (harness, kvsim, and context-eval bundles, validated against the result model); TypeScript types are
  generated from the result JSON Schema, so schema drift fails the build.

## Generated state

`artifacts/`, `results/`, `release/`, and `dataset/` (except the small frozen
`dataset/processed/hotpot_portable/` fixture) contain local or generated state.
`dashboard/public/data/` is the exception: the exported result snapshot is committed, and the `pages`
workflow publishes the dashboard built from it to GitHub Pages. Source, tests, configs, experiment specs, and public documentation remain in Git.
