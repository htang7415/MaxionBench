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

The Go gateway needs the Go version in `gateway/go.mod`; the dashboard needs Node 24.

Keep the virtual environment on an internal disk. On exFAT or other external volumes, macOS writes
`._*` AppleDouble files that break Python imports and pytest collection, and imports run up to 10x
slower. Remove stray metadata before testing or committing:

```bash
find . -name '._*' -not -path './.git/*' -not -path './dashboard/node_modules/*' -delete
```

Datasets for v0.3 experiments are downloaded and checked against pinned SHA-256 values:

```bash
python -m maxionbench.datasets.sources fetch     # into dataset/v03/ (git-ignored)
python -m maxionbench.datasets.sources verify
```

## Validate changes

These mirror the `v03-ci` workflow:

```bash
python -m ruff check maxionbench scripts tests
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

Generated files under `artifacts/`, `results/`, `release/`, and `dashboard/public/data/` are not
source files and should not be committed. Experiments that call paid APIs must go through the spend
ledger (`harness.budget` / `eval.batch.metered`) and never run in CI.

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
