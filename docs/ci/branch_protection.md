# CI and branch protection

## Workflow

All checks run in one workflow, `.github/workflows/v03_ci.yml` (name `v03-ci`), on pushes to `main`
and `v0.3-harness`, on pull requests, and on manual dispatch. It makes no paid API calls and uses no
secrets.

| Job | What it checks |
| --- | --- |
| `python` | Ruff; result JSON Schema is current (`python -m maxionbench.harness schema --check`); pytest (Python 3.12, hash-locked dependencies; Go installed so gateway end-to-end tests run) |
| `go` | `gofmt`, `go vet`, `go test -race ./...` in `gateway/` |
| `dashboard` | `npm ci`; generated TypeScript types match the result schema; Vitest; production build |
| `perf-smoke` | Runs `experiments/ci_smoke_sim.yaml` (llm-d-inference-sim) and `experiments/ci_smoke_llamacpp.yaml` (pinned llama.cpp CPU build, Qwen3-0.6B Q8_0, both SHA-256 verified), then `python -m maxionbench.harness.perf_gate ci/perf_baseline.yaml` |

pytest deselects two `tests/test_repo_hygiene.py` checks that predate v0.3 and assert that
`AGENTS.md` and local-only `docs/` files are tracked, which the project deliberately does not do.
Playwright tests (`cd dashboard && npm run e2e`) need exported results and run locally only.

## Performance gate

`ci/perf_baseline.yaml` bounds each experiment's cell means. The inference simulator has a fixed
latency model, so its bounds are tight (TTFT p50 790–970 ms): drift means the harness's load
generation or timing changed. CPU inference on shared runners varies, so llama.cpp has a floor of
6 tokens/s and a 5 s TTFT p50 ceiling, which only a roughly 2× regression crosses. Any request error
or failed trial fails the gate. Recalibrate by running both experiments on the target runner and
recording the observed values in the file's comments.

## Branch protection (recommended)

`main` is currently not protected. To protect it, require a pull request before merging and require
these status checks:

- `v03-ci / python`
- `v03-ci / go`
- `v03-ci / dashboard`
- `v03-ci / perf-smoke`

These are the defaults of the verifier:

```bash
maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json
```

It uses `GITHUB_TOKEN` (or `--token`) and exits `0` when every required check is configured and `2`
when some are missing. If a job is renamed, update `DEFAULT_REQUIRED_CHECKS` in
`maxionbench/tools/verify_branch_protection.py`, this document, and the pull request template together.

## Retired in v0.3

`report-preflight` and `branch-protection-drift` (and the `snapshot-required-checks` command that kept
them in sync) were removed: both failed on every run after the dependency lock moved to numpy 2.5,
which needs Python 3.12, and the drift check targeted protection that `main` does not have.
