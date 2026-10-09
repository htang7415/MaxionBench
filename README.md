# MaxionBench

**[View the results dashboard →](https://htang7415.github.io/MaxionBench/)** · [Architecture](ARCHITECTURE.md)

MaxionBench measures how to serve LLM agents efficiently. It runs real engines (vLLM on Apple GPUs via
`vllm-metal`, llama.cpp), llm-d request scheduling, and a Go AI gateway in front of a local fleet and the
Gemini API, and reports accuracy, latency, and cost with confidence intervals.

The current release, **v0.5**, adds cache-aware context management to the gateway. Each agent session's
prompt stays append-only, so the engine's prefix cache keeps hitting. The gateway trims the prompt only when
it passes a token budget, or when the cache is cold anyway.

## Highlights

| Result | Setting |
| --- | --- |
| **−14% to −39%** prefill recompute, in all 4 paired runs | Gateway on replayed GitHub Copilot traffic, Qwen3-8B on vllm-metal |
| **17% → 92%** of requests under 5 s to first token | Same, on the overloaded run |
| **−50%** prompt tokens, no accuracy loss (52% vs 44%) | Gateway in front of Gemini, 100 paired BrowseComp-Plus tasks |
| **up to 2.6×** more prefill when agents trim context naively | Copilot sessions replayed under each context policy |
| **57k** median prompt, **48%** tool output | 9.1M GitHub Copilot coding-agent calls |

## How it works

```text
agent ──► Go gateway ──► llm-d (EPP + Envoy) ──► vLLM / llama.cpp
          │ append-only views per session; trim past a budget or when cold
          └─ overflow / failover ──► Gemini API (hard spend cap)
```

| Path | What it does |
| --- | --- |
| `gateway/` | Go gateway: routing, overflow, failover, spend cap, context management, metrics, tracing |
| `maxionbench/harness` | Experiment specs, load generation, 95% CIs, provenance |
| `maxionbench/agents` | Agent loops and context policies (`full`, `truncate`, `window`, `mask`, `summarize`, cache-aware) |
| `maxionbench/kvsim` | KV-cache simulator and live replays onto real engines |
| `maxionbench/eval` | Accuracy and cost studies with a calibrated LLM judge and paired tests |
| `dashboard/` | TypeScript results site (the link above) |

## Quickstart

Requires Python 3.12, Go, and Node 24.

```bash
python3.12 -m venv ~/.venvs/maxionbench && . ~/.venvs/maxionbench/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -e ".[agents]"                          # MCP agent server, BM25 search

python -m maxionbench.datasets.sources fetch                    # pinned datasets, SHA-256 verified
python -m maxionbench.harness run experiments/e2_prefix_caching.yaml
cd dashboard && npm ci && npm run data && npm run dev           # view your results
```

Paid-API runs read the key only at runtime (`GEMINI_API_KEY`). Before sending a request, they reserve its
worst-case cost against a hard cap (`configs/pricing/gemini.yaml`).

## Releases

- **v0.5** — context management in the Go gateway (K9 on vllm-metal, C2 on Gemini)
- **v0.4** — context policies for agents: Copilot traces, accuracy and API cost, serving cost
- **v0.3** — serving harness: engines, prefix caching, llm-d scheduling, hybrid overflow
- **v0.1–v0.2** — an earlier vector-database benchmark, kept at its tags

## License

See [LICENSE](LICENSE).
