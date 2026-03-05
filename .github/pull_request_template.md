## Summary

Describe the change and why it is needed.

## Validation

- [ ] `report-preflight / report_preflight` passed
- [ ] `report-preflight / legacy_migration_path` passed
- [ ] `report-preflight / legacy_resource_profile_path` passed
- [ ] `report-preflight / legacy_ground_truth_metadata_path` passed
- [ ] `branch-protection-drift / verify_branch_protection` passed (if enforced)

## Artifact/Report Notes

- [ ] If artifact schema/report paths changed, I ran:
  - `maxionbench validate --input artifacts/runs --json`
  - `maxionbench migrate-stage-timing --input artifacts/runs --dry-run` (if needed)
  - `maxionbench report --input artifacts/runs --mode milestones --out artifacts/figures/milestones`

## References

- Branch protection policy: `docs/ci/branch_protection.md`
