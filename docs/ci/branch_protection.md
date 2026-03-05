# Branch Protection Policy: `main`

This repository relies on CI status checks to protect report-generation correctness and artifact preflight guarantees.

## Required settings (GitHub)

For branch `main`, configure branch protection with:

1. `Require a pull request before merging`
2. `Require status checks to pass before merging`

Required checks:

- `report-preflight / conformance_readiness_gate`
- `report-preflight / report_preflight`
- `report-preflight / legacy_migration_path`
- `report-preflight / legacy_resource_profile_path`
- `report-preflight / legacy_ground_truth_metadata_path`

Optional (recommended once token permissions are stable):
- `branch-protection-drift / verify_branch_protection`

## Why these checks are required

- `conformance_readiness_gate` verifies pre-run readiness policy wiring:
  - generates `artifacts/conformance/conformance_matrix.csv`
  - validates behavior-card coverage and conformance-matrix adapter coverage via `maxionbench verify-engine-readiness`
  - preserves a CI artifact trail for readiness gating inputs
- `report_preflight` verifies the normal path:
  - smoke benchmark run
  - artifact validation (`maxionbench validate`)
  - report generation (`maxionbench report`)
- `legacy_migration_path` verifies the failure/recovery path:
  - report fails on legacy artifacts missing stage timing columns
  - migration (`maxionbench migrate-stage-timing`) backfills columns
  - report succeeds after migration
- `legacy_resource_profile_path` verifies RHU schema enforcement:
  - strict validation fails when RHU resource columns/metadata are missing
  - `--legacy-ok` surfaces explicit warnings for local inspection
  - report generation fails with RHU remediation hint for legacy artifacts
- `legacy_ground_truth_metadata_path` verifies ground-truth provenance enforcement:
  - strict validation fails when `ground_truth_*` metadata fields are missing/invalid
  - `--legacy-ok` surfaces warning diagnostics for local inspection
  - report generation fails with ground-truth remediation hint for legacy artifacts

## Maintenance note

If workflow/job names change, update this policy doc and `.github/pull_request_template.md` in the same PR.

## Automatic policy-sync guards

These consistency checks are enforced in CI and should pass in the same PR as any workflow/check rename:

- `tests/test_branch_protection_policy_sync.py`
  - `report_preflight.yml` jobs <-> required check contexts in this doc
  - `report_preflight.yml` jobs <-> required check checklist entries in `.github/pull_request_template.md`
  - `report_preflight.yml` jobs <-> `maxionbench.tools.verify_branch_protection.DEFAULT_REQUIRED_CHECKS`
  - `branch_protection_drift.yml --required-check ...` <-> `DEFAULT_REQUIRED_CHECKS`
- `tests/test_branch_protection_drift_workflow.py`
  - drift workflow command shape and required-check arguments
- `tests/test_report_preflight_workflow.py`
  - preflight workflow structure and required legacy safety-path checks
- `tests/test_report_figure_policy_sync.py`
  - report figure IDs/style pins and prompt alignment for milestone/final exports
- CI artifact snapshot command:
  - `maxionbench snapshot-required-checks --output artifacts/ci/required_checks_snapshot.json --strict --json`
  - writes `artifacts/ci/required_checks_snapshot.json` for auditable required-check context parity

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
