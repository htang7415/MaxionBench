## Summary

Describe the change and why it is needed.

## Validation

- [ ] `python -m ruff check maxionbench scripts tests`
- [ ] `python -m pytest -q`
- [ ] `python scripts/build_package.py`
- [ ] `report-preflight / conformance_readiness_gate` passed
- [ ] `report-preflight / report_preflight` passed
- [ ] `branch-protection-drift / verify_branch_protection` passed (if enforced)

## Artifact/Report Notes

- [ ] If artifact schema/report paths changed, I ran:
  - `maxionbench validate --input artifacts/runs --strict-schema --json`
  - `maxionbench migrate-stage-timing --input artifacts/runs --dry-run` (if needed)
  - `maxionbench report --input artifacts/runs --mode maxionbench --out artifacts/figures/final`

## References

- Branch protection policy: `docs/ci/branch_protection.md`
