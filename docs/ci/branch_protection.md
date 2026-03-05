# Branch Protection Policy: `main`

This repository relies on CI status checks to protect report-generation correctness and artifact preflight guarantees.

## Required settings (GitHub)

For branch `main`, configure branch protection with:

1. `Require a pull request before merging`
2. `Require status checks to pass before merging`

Required checks:

- `report-preflight / report_preflight`
- `report-preflight / legacy_migration_path`

Optional (recommended once token permissions are stable):
- `branch-protection-drift / verify_branch_protection`

## Why these checks are required

- `report_preflight` verifies the normal path:
  - smoke benchmark run
  - artifact validation (`maxionbench validate`)
  - report generation (`maxionbench report`)
- `legacy_migration_path` verifies the failure/recovery path:
  - report fails on legacy artifacts missing stage timing columns
  - migration (`maxionbench migrate-stage-timing`) backfills columns
  - report succeeds after migration

## Maintenance note

If workflow/job names change, update this policy doc and `.github/pull_request_template.md` in the same PR.

## Optional drift check command

You can verify current GitHub branch protection status via API:

```bash
maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --json
maxionbench verify-branch-protection --repo <owner>/<repo> --branch main --include-drift-check --json
```

Notes:
- Uses `GITHUB_TOKEN` by default (or pass `--token`).
- Returns exit code `0` when required checks are present, `2` when checks are missing.

## Automated drift workflow

Workflow:
- `.github/workflows/branch_protection_drift.yml`

Behavior:
- runs on schedule and on manual dispatch
- executes `maxionbench verify-branch-protection` for `main`
- uploads `branch_protection_summary.json` as an artifact

Auth note:
- workflow prefers `BRANCH_PROTECTION_TOKEN` secret (recommended: repo-admin PAT)
- falls back to `github.token`; if insufficient for branch-protection API access, configure `BRANCH_PROTECTION_TOKEN`
