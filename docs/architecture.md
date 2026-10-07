# Architecture

MaxionBench is a Python package plus a Go gateway and a TypeScript dashboard. The earlier
vector-database benchmark lives at the `v0.1` tag and the `v0.2-cpu-infra` branch.

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
  it is sent, then commits provider-reported usage. Billed output is visible plus thinking tokens
  (Gemini's OpenAI-compatible usage reports thinking only in `total_tokens`). HTTP error responses
  are not billed; timeouts are charged their estimate.
- **Keys** are loaded only at runtime via `harness.secrets`; results record key presence, not value.

## Datasets and evaluation

- `maxionbench/datasets/sources.py` downloads and derives the v0.3 datasets into `dataset/v03/`, and
  `manifests/v03.yaml` pins every file by SHA-256 (CRAG-500, BEIR SciFact/FiQA, a ShareGPT sample,
  the Azure LLM trace 2024, BFCL v3). Loaders verify a file before reading it.
- `maxionbench/eval/e5.py` runs one model on QA with provided context (CRAG search snippets; HotpotQA
  gold paragraphs plus distractors), BFCL v3 single-turn tool calls, and agentic HotpotQA. Requests run
  at concurrency 1; items are split into seeded shards that act as repeats in the result schema.
- `maxionbench/eval/e6.py` compares Gemini implicit caching, explicit `cachedContents`, and the inline
  Batch API on one shared-document workload.
- Graders (`maxionbench/graders/`): BFCL AST checker; EM/F1 and CRAG's three-way score; agent task
  success; an LLM judge whose rubric is calibrated against `graders/calibration/qa_judge_v1.jsonl`.
- Agents (`maxionbench/agents/`): an MCP stdio server exposing `search`/`read` over the HotpotQA
  corpus, and an agent loop that replays the model's own assistant message each turn (Gemini 3 thought
  signatures must come back unchanged).

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
- Prometheus metrics (requests, route decisions, in-flight, time to first byte, spend, budget,
  predicted local wait) and OpenTelemetry tracing: W3C `traceparent` is always propagated to the
  fleet and the remote; spans are exported over OTLP only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

## Observability and dashboard

- `deploy/observability/`: OpenTelemetry collector → Jaeger, Prometheus scraping the gateway, the
  llm-d EPP, vLLM, and llama.cpp, and Grafana with a provisioned serving dashboard. Every port binds
  to 127.0.0.1.
- `dashboard/`: a static Vite + React + Recharts site. `npm run data` exports the latest complete run
  of each published experiment (validated against the result model); TypeScript types are generated
  from the result JSON Schema, so schema drift fails the build.

## Generated state

`artifacts/`, `results/`, `release/`, and `dataset/` (except the small frozen
`dataset/processed/hotpot_portable/` fixture) contain local or generated state, as does
`dashboard/public/data/`. Source, tests, configs, experiment specs, and public documentation remain in Git.
