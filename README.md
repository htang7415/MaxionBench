# MaxionBench

**[Results dashboard](https://htang7415.github.io/MaxionBench/)** · [Architecture](docs/architecture.md) · [Contributing](docs/contributing.md) · [Security](docs/security.md) · [Releases](https://github.com/htang7415/MaxionBench/tags)

MaxionBench is a reproducible harness for LLM serving and agents. It runs real engines (vLLM on Apple
GPUs via `vllm-metal`, llama.cpp), real llm-d request scheduling, and a Go AI gateway in front of a local
fleet and the Gemini API, then measures accuracy, latency, and cost with confidence intervals and full
provenance.

The current release, **v0.5**, puts cache-aware context management for agents into the gateway: each
agent session's prompt stays append-only so the engine's prefix cache keeps hitting, and is trimmed only
when it grows past a budget or the cache is cold anyway.

## Highlights

| Result | Setting |
| --- | --- |
| **−14% to −39%** prefill recompute, in all 4 paired runs | Gateway context management on replayed GitHub Copilot traffic, Qwen3-8B on vllm-metal |
| **17% → 92%** of requests under 5 s to first token | Same, on the overloaded run |
| **−50%** prompt tokens, no accuracy loss (52% vs 44%) | Gateway in front of Gemini, 100 paired BrowseComp-Plus tasks |
| **up to 2.6×** more prefill when an agent trims its context naively | Copilot sessions replayed under each context policy |
| **57k** median prompt, **48%** tool output | 9.1M GitHub Copilot coding-agent calls |

## How it works

```text
agent ──► Go gateway ──────────────► llm-d (EPP + Envoy) ──► vLLM / llama.cpp replicas
          │ context manager:                                  (prefix cache, KV events)
          │   append-only per session,
          │   trim past a budget or when cold
          └─ overflow / failover ──► Gemini API (hard spend cap, shared ledger)

harness: YAML spec ──► trials (matrix × repeats) ──► results.json (95% CIs, provenance) ──► dashboard
```

| Component | What it does |
| --- | --- |
| [`gateway/`](gateway) (Go) | OpenAI-compatible proxy: local-first routing, SLO-aware overflow to Gemini, failover, spend reservations, cache-aware context management, Prometheus metrics, OpenTelemetry tracing |
| [`maxionbench/harness`](maxionbench/harness) | Experiment specs, open- and closed-loop load generation measured from scheduled arrivals, CIs, provenance, generated result schema |
| [`maxionbench/agents`](maxionbench/agents) | Context policies (`full`, `truncate`, `window`, `mask`, `summarize`, cache-aware variants), MCP and BrowseComp-Plus agent environments |
| [`maxionbench/kvsim`](maxionbench/kvsim) | KV-cache simulator for agent traces, and live replays through llm-d and the gateway onto real engines |
| [`maxionbench/eval`](maxionbench/eval) | Accuracy and cost studies with a calibrated LLM judge, strict regrading, Holm-corrected McNemar tests |
| [`dashboard/`](dashboard) | TypeScript results site, published at the link above |
| [`deploy/`](deploy) | llm-d without Kubernetes, and an observability stack (OTel collector, Jaeger, Prometheus, Grafana) |

## Installation

Requires Python 3.12, Go (version in `gateway/go.mod`), and Node 24. GPU experiments on a Mac use vLLM
with the `vllm-metal` plugin from a separate environment.

```bash
python3.12 -m venv ~/.venvs/maxionbench && . ~/.venvs/maxionbench/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -e ".[agents]"
```

## Quickstart

```bash
python -m maxionbench.datasets.sources fetch                      # pinned datasets, SHA-256 verified
python -m maxionbench.harness run experiments/e2_prefix_caching.yaml
python -m maxionbench.harness compare <run_dir_a> <run_dir_b>    # CI overlap between runs
cd dashboard && npm ci && npm run data && npm run dev             # browse your results
```

Paid-API experiments read the key only at runtime (`GEMINI_API_KEY` or a git-ignored key file) and reserve
each request's worst-case cost against a hard cap (`configs/pricing/gemini.yaml`) before sending it.

## Results

### v0.5 — context management in the gateway

The gateway's context manager (`gateway/internal/ctxmgr`, `context:` in its config) keys each agent session by
`prompt_cache_key`, else by its system prompt and task. Sessions that share a key keep separate state, and a
request continues the session whose history it extends. The view sent upstream only grows until it passes a
token budget, or until the client's history is new or rewritten. After a trim, the view must grow by
`min_growth` tokens before the next trim. The Go implementation produces the same views as
`agents/context.py` on a fixture generated from Python. Each decision is reported in a response header,
Prometheus counters, and trace attributes.

**K9: Copilot traffic through the gateway onto a real engine.** Copilot coding-agent sessions (a Saturday and
a Tuesday with ~5× the traffic) replayed as full chat histories in real trace time, message text replaced
by filler at 1/16 scale, through the gateway onto one vllm-metal replica of Qwen3-8B (MLX 4-bit, Apple M4,
24.7k-token KV cache). Two seeded session draws per day; every arm replays the same schedule as `off`.

| Recomputed prefill tokens per request | Sat, draw 1 | Sat, draw 2 | Tue, draw 1 | Tue, draw 2 |
| --- | --- | --- | --- | --- |
| `off` (full history) | 290 | 197 | 301 | 303 |
| `window+cache` (8 exchanges) | **−14%** | **−30%** | **−39%** | **−27%** |
| `mask+cache` (4 exchanges, `min_growth` 25k) | −10% | +37% | −31% | −19% |
| `mask+cache` + trim after 120 s idle | −9% | +36% | −26% | −20% |
| `mask+cache` without `min_growth` | +2% | +84% | −25% | −12% |

Most of the gain comes from trimming when a session first appears or the agent rewrites its own history,
when the prefix is not cached anyway; the budget itself fired on under 2% of requests. On the heavier
Saturday draw, full histories overloaded the server (median TTFT 80 s, 17% of requests under 5 s);
`window+cache` kept it under capacity (1.0 s, 92%). Masking keeps every assistant message, so its views stay
large on long sessions. Recomputed tokens repeat exactly across reruns; TTFT depends on host load.

**C2: the gateway in front of Gemini.** The C1 agent (below) sends its full history; the gateway applies
`window+cache` (last 2 exchanges, budget 48k tokens, `min_growth` 16k) before Gemini. 100 tasks, paired with
`full` on the same tasks.

| 100 tasks | Judge | Strict | Billed cost / task | Prompt tokens / task | Cost / correct | Cached |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 44% | 38% | $0.035 | 287k | $0.081 | 66% |
| gateway `window+cache` | 52% | 48% | −12% [−36%, +13%] | **−50%** [−80%, −20%] | $0.060 | 32% |
| difference (wins/losses, p, Holm p) | +8 (17/9, 0.17, 0.51) | +10 (17/7, 0.064, 0.19) | | | | |

Trimming in the gateway halved billed prompt tokens with no accuracy loss; the accuracy gain is not
significant at 100 tasks. Cost fell less than tokens because each trim gives up Gemini's implicit cache. On
C1's tasks the gateway matched the in-agent `window+cache` policy (56% vs 56%). Gemini rejects
`prompt_cache_key`, so the gateway strips it before forwarding.

### v0.4 — context policies for agents

What should an agent send the model each step: its whole history, or a trimmed view?

| Policy | What the model sees |
| --- | --- |
| `full` | the whole history (baseline) |
| `truncate` | every tool result cut to its first 2k tokens |
| `window` | only the last N exchanges |
| `mask` | tool results older than the last N exchanges replaced by a placeholder |
| `summarize` | above a token trigger, older exchanges replaced by an LLM-written summary |
| `window+cache`, `mask+cache` | append-only view; the base policy is applied only when the view passes a token budget |

**Production agents** (GitHub Copilot coding-agent traces, June 1–7 2026: 301k sessions, 9.1M calls). Median
prompt 57k tokens, 76% of calls above 30k; tool output is 48% of prompt tokens. Copilot rarely cuts context
(8.6% of sessions, at ~130k tokens), and the call after a cut is 27% cached versus 89% for steady calls. Cache
hits fall with the pause before a call: 93% under 10 s, 77% at 1–5 min, 23% at 5–60 min; the median pause
between turns is 170 s.

**C1: accuracy and API cost** (Gemini 3.5 Flash-Lite agent on BrowseComp-Plus, 50 tasks × 7 policies, each task
searching ~800 web pages; difference vs `full` on the same tasks, 95% CI). Judge: the calibrated Gemini judge;
strict: string match against the gold answer (they agree on 93% of answers); p: exact McNemar vs `full`,
Holm-adjusted over the six policies.

| Policy | Judge | Δ judge | p (Holm) | Strict | Δ strict | p (Holm) | Cost / task | Δ cost | Cost / correct | Cached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 46% | – | – | 38% | – | – | $0.032 | – | $0.069 | 64% |
| `truncate` | 40% | −6 pts [−20, +8] | 1.0 | 36% | −2 | 1.0 | $0.013 | **−58%** [−93%, −22%] | **$0.033** | 68% |
| `window` | 58% | +12 [−1, +25] | 0.73 | 54% | +16 | 0.13 | $0.031 | −2% | $0.053 | 12% |
| `mask` | 52% | +6 [−6, +18] | 1.0 | 48% | +10 | 0.58 | $0.024 | −25% | $0.046 | 0% |
| `summarize` | **66%** | **+20 [+7, +33]** | **0.04** | 54% | +16 | 0.13 | $0.039 | +24% | $0.059 | 17% |
| `window+cache` | 56% | +10 [−3, +23] | 0.91 | 48% | +10 | 0.58 | $0.022 | −29% | **$0.040** | 53% |
| `mask+cache` | 56% | +10 [−4, +24] | 0.91 | 50% | +12 | 0.58 | $0.026 | −17% | $0.047 | 52% |

Five of six trimming policies were at least as accurate as `full` under both gradings; at 50 tasks only
`summarize` under the judge is significant after correction. Policies that rewrite earlier messages lost
Gemini's implicit cache (`mask` 0% cached); the cache-aware variants kept half of it and cost 17–29% less.

**K6–K8: serving cost** (Copilot sessions replayed under each policy as prefix-chained KV blocks; prefill
recomputed vs `full`). K6: offline KV simulator, 4 replicas, llm-d-style prefix+load routing, 3 repeats. K7:
live through the llm-d EPP + Envoy over 4 inference-sim workers at the tight capacity. K8: through llm-d onto
one vllm-metal replica of Qwen3-0.6B, prompts at 1/32 scale.

| Policy | Prompt size | K6, 512k KV/replica | K6, 1M | K6, 64M | K7 live, 512k | K7 TTFT p99 | K8 vllm-metal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 68k | – | – | – | – | 1.49 s | – |
| `truncate` | 56k | −37% | −28% | −20% | −60% | 0.77 s | −46% |
| `window` | 21k | +20% | +134% | **+162%** | +34% | 0.84 s | +71% |
| `mask` | 36k | −14% | +45% | +64% | −16% | 0.70 s | +13% |
| `summarize` | 36k | −15% | +12% | +11% | −35% | 1.11 s | −6% |
| `window+cache` | 32k | **−53%** | −10% | +5% | −58% | 0.58 s | −36% |
| `mask+cache` | 44k | −46% | −14% | −2% | −58% | 0.61 s | −41% |

With ample KV memory, policies that rewrite the prompt make the server recompute more despite sending 50–70%
fewer tokens; under KV pressure, smaller contexts avoid evictions. The cache-aware variants are the only
policies that win under pressure without losing much otherwise. Live hit ratios match the simulator within
~3 points.

### v0.3 — serving on one Mac

- **Engines:** on the same Qwen3-0.6B file, llama.cpp wins single-stream (107 vs 79 tok/s); vLLM decodes
  25–35% faster per token from concurrency 4.
- **Prefix caching:** vLLM automatic prefix caching on multi-turn RAG cut TTFT p50 from 1,014 to 375 ms
  (3.9× goodput).
- **llm-d scheduling:** prefix + load scoring beat prefix-only (−28% goodput) and load-only (−59%) profiles
  on simulated workers.
- **Hybrid serving:** at concurrency 24, overflow from the local fleet to Gemini lifted goodput 58% at 57%
  of the all-remote cost.

## Reproduce

```bash
python -m maxionbench.datasets.sources fetch --group copilot --group copilot_traces --group browsecomp_plus
python -m maxionbench.eval.copilot_characterize --jobs 3                     # production characterization
python -m maxionbench.eval.context_eval experiments/c1_context_policies.yaml  # Gemini; paid, resumable
python -m maxionbench.eval.context_regrade artifacts/context_eval/<run>       # strict grading, McNemar + Holm
python -m maxionbench.kvsim experiments/k6_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k7_llmd_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.live experiments/k8_vllm_metal_copilot_context_policies.yaml --out artifacts/kvsim
python -m maxionbench.kvsim.gateway_replay experiments/k9_gateway_context_8b.yaml --out artifacts/kvsim
python -m maxionbench.eval.context_eval experiments/c2a_gateway_context.yaml   # Gemini; paid, resumable
python -m maxionbench.eval.context_eval experiments/c2b_gateway_context.yaml
python -m maxionbench.eval.gateway_accuracy <c1 run> <c2a run> <c2b run> --out gateway_accuracy.json
```

BrowseComp-Plus text must not be published: result files hold task ids and numbers only, and model answers
stay in a local file. CI ([`v03_ci.yml`](.github/workflows/v03_ci.yml)) runs lint, Python, Go, and TypeScript
tests, schema and type drift checks, and a CPU performance gate, with no paid API calls.

## Repository layout

| Path | Purpose |
| --- | --- |
| `maxionbench/` | Harness, evaluation, graders, agents, datasets, KV-cache simulator, runtime metadata |
| `gateway/` | Go AI gateway |
| `dashboard/` | TypeScript results site; `public/data/` holds the published result snapshot (`npm run data`) |
| `experiments/` | Specs: v0.3 E1–E6, v0.4 C1 and K1–K8, v0.5 K9 and C2, CI smoke runs |
| `configs/` | API pricing and the spend cap |
| `deploy/` | llm-d without Kubernetes, the observability stack |
| `ci/` | Performance-gate bounds |
| `docs/` | Architecture, contributing, security, CI policy |
| `tests/` | Python tests |

Earlier versions — a vector-database benchmark for agent memory — are at the `v0.1` and `v0.2` tags.

## License

See [LICENSE](LICENSE).
