# Contributing

## Setup

Python 3.12 is required (the hash-locked dependencies pin numpy 2.5).

```bash
python3.12 -m venv ~/.venvs/maxionbench
. ~/.venvs/maxionbench/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -e ".[agents]"       # MCP agent server and BM25 search
```

The Go gateway needs the Go version in `gateway/go.mod`; the dashboard needs Node 24. Experiments on
the Mac GPU use vLLM with the `vllm-metal` plugin from a separate environment (default
`~/.venv-vllm-metal/bin/vllm`, set `vllm:` in a target to override).

Keep the virtual environment on an internal disk. On exFAT or other external volumes, macOS writes
`._*` AppleDouble files that break Python imports and pytest collection, and imports run up to 10x
slower. Remove stray metadata before testing or committing:

```bash
find . -name '._*' -not -path './.git/*' -not -path './dashboard/node_modules/*' -delete
```

Datasets are downloaded and checked against pinned SHA-256 values:

```bash
python -m maxionbench.datasets.sources fetch                    # all groups, into dataset/v03/ (git-ignored)
python -m maxionbench.datasets.sources fetch --group copilot     # one group (repeat --group for more)
python -m maxionbench.datasets.sources verify
```

## Validate changes

These mirror the `v03-ci` workflow:

```bash
python -m ruff check maxionbench tests
python -m maxionbench.harness schema --check
python -m pytest -q
(cd gateway && gofmt -l . && go vet ./... && go test -race ./...)
(cd dashboard && npm ci && npm run types && npx vitest run && npm run build)
```

Optional local checks: `cd dashboard && npm run data && npm run e2e` (Playwright over exported
results), and the CPU performance smoke runs in `experiments/ci_smoke_*.yaml` followed by
`python -m maxionbench.harness.perf_gate ci/perf_baseline.yaml artifacts/ci/*/results.json`.

Keep changes focused. Add tests for behavior changes. When the result dataclasses change, regenerate
the schema (`python -m maxionbench.harness schema --write`) and the dashboard types (`npm run types`).

Generated files under `artifacts/`, `results/`, and `release/` are not source files and should not be
committed. `dashboard/public/data/` is the published result snapshot: refresh it with `npm run data` after
a new published run and commit it; the `pages` workflow redeploys the dashboard. Experiments that call paid APIs must go through the spend
ledger (`harness.budget`: per-request reservations, or `eval.batch.metered` for a batch run alone on the
cap) and never run in CI.

After changing dependencies, regenerate the lock file with Python 3.12:

```bash
python -m piptools compile \
  --extra dev \
  --allow-unsafe \
  --generate-hashes \
  --strip-extras \
  --output-file requirements-dev.lock \
  pyproject.toml
```

## CI and branch protection

### Workflow

All checks run in one workflow, `.github/workflows/v03_ci.yml` (name `v03-ci`), on pushes to `main`,
on pull requests, and on manual dispatch. It makes no paid API calls and uses no
secrets.

| Job | What it checks |
| --- | --- |
| `python` | Ruff; result JSON Schema is current (`python -m maxionbench.harness schema --check`); pytest (Python 3.12, hash-locked dependencies; Go installed so gateway end-to-end tests run) |
| `go` | `gofmt`, `go vet`, `go test -race ./...` in `gateway/` |
| `dashboard` | `npm ci`; generated TypeScript types match the result schema; Vitest; production build |
| `perf-smoke` | Runs `experiments/ci_smoke_sim.yaml` (llm-d-inference-sim) and `experiments/ci_smoke_llamacpp.yaml` (pinned llama.cpp CPU build, Qwen3-0.6B Q8_0, both SHA-256 verified), then `python -m maxionbench.harness.perf_gate ci/perf_baseline.yaml` |

Playwright tests (`cd dashboard && npm run e2e`) need exported results and run locally only.

### Performance gate

`ci/perf_baseline.yaml` bounds each experiment's cell means. The inference simulator has a fixed
latency model, so its bounds are tight (TTFT p50 790–970 ms): drift means the harness's load
generation or timing changed. CPU inference on shared runners varies, so llama.cpp has a floor of
6 tokens/s and a 5 s TTFT p50 ceiling, which only a roughly 2× regression crosses. Any request error
or failed trial fails the gate. Recalibrate by running both experiments on the target runner and
recording the observed values in the file's comments.

### Branch protection (recommended)

`main` is currently not protected. To protect it, require a pull request before merging and require
these status checks:

- `v03-ci / python`
- `v03-ci / go`
- `v03-ci / dashboard`
- `v03-ci / perf-smoke`

If a job is renamed, update this document and the pull request template together.

### Retired in v0.3

`report-preflight` and `branch-protection-drift` (and the `snapshot-required-checks` command that kept
them in sync) were removed: both failed on every run after the dependency lock moved to numpy 2.5,
which needs Python 3.12, and the drift check targeted protection that `main` does not have.
