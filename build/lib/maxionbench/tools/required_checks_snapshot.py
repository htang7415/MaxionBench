"""Generate an auditable snapshot of required CI check contexts."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import yaml

from maxionbench.tools.verify_branch_protection import DEFAULT_REQUIRED_CHECKS, OPTIONAL_REQUIRED_CHECKS


def build_required_checks_snapshot(
    *,
    report_workflow_path: Path,
    drift_workflow_path: Path,
    branch_protection_doc_path: Path,
    pr_template_path: Path,
) -> dict[str, Any]:
    report_jobs = sorted(_report_preflight_job_names(report_workflow_path))
    contexts_from_jobs = sorted([f"report-preflight / {name}" for name in report_jobs])
    contexts_from_drift = sorted(_drift_required_checks(drift_workflow_path))
    contexts_from_defaults = sorted(set(DEFAULT_REQUIRED_CHECKS))
    contexts_from_doc = sorted(_branch_protection_doc_required_checks(branch_protection_doc_path))
    contexts_from_pr = sorted(_pr_template_checklist_checks(pr_template_path))

    def _missing(base: list[str], candidate: list[str]) -> list[str]:
        return sorted(set(base) - set(candidate))

    def _extra(base: list[str], candidate: list[str]) -> list[str]:
        return sorted(set(candidate) - set(base))

    missing_vs_defaults = _missing(contexts_from_jobs, contexts_from_defaults)
    extra_vs_defaults = _extra(contexts_from_jobs, contexts_from_defaults)
    missing_vs_drift = _missing(contexts_from_jobs, contexts_from_drift)
    extra_vs_drift = _extra(contexts_from_jobs, contexts_from_drift)
    missing_vs_doc = _missing(contexts_from_jobs, contexts_from_doc)
    extra_vs_doc = _extra(contexts_from_jobs, contexts_from_doc)
    missing_vs_pr = _missing(contexts_from_jobs, contexts_from_pr)
    extra_vs_pr = _extra(contexts_from_jobs, contexts_from_pr)
    extra_report_preflight_vs_pr = sorted(
        context for context in extra_vs_pr if context.startswith("report-preflight / ")
    )
    extra_non_report_preflight_vs_pr = sorted(
        context for context in extra_vs_pr if not context.startswith("report-preflight / ")
    )
    unexpected_optional_vs_pr = sorted(
        context
        for context in extra_non_report_preflight_vs_pr
        if context not in set(OPTIONAL_REQUIRED_CHECKS)
    )

    checks = {
        "jobs_vs_defaults": not missing_vs_defaults and not extra_vs_defaults,
        "jobs_vs_drift_workflow": not missing_vs_drift and not extra_vs_drift,
        "jobs_vs_branch_protection_doc": not missing_vs_doc and not extra_vs_doc,
        # PR template may include optional checks from other workflows (for example
        # `branch-protection-drift / verify_branch_protection`) as long as all required
        # report-preflight contexts are present, there are no stale report-preflight
        # contexts, and optional contexts are in the allowlist.
        "jobs_vs_pr_template": (
            (not missing_vs_pr)
            and (not extra_report_preflight_vs_pr)
            and (not unexpected_optional_vs_pr)
        ),
        "pr_template_optional_contexts_valid": not unexpected_optional_vs_pr,
    }

    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "report_preflight_workflow": str(report_workflow_path.resolve()),
        "branch_protection_drift_workflow": str(drift_workflow_path.resolve()),
        "branch_protection_doc": str(branch_protection_doc_path.resolve()),
        "pr_template": str(pr_template_path.resolve()),
        "report_preflight_job_names": report_jobs,
        "required_check_contexts": {
            "from_report_preflight_jobs": contexts_from_jobs,
            "from_verify_branch_protection_defaults": contexts_from_defaults,
            "from_branch_protection_drift_workflow": contexts_from_drift,
            "from_branch_protection_doc": contexts_from_doc,
            "from_pr_template": contexts_from_pr,
        },
        "diff": {
            "missing_vs_defaults": missing_vs_defaults,
            "extra_vs_defaults": extra_vs_defaults,
            "missing_vs_drift_workflow": missing_vs_drift,
            "extra_vs_drift_workflow": extra_vs_drift,
            "missing_vs_branch_protection_doc": missing_vs_doc,
            "extra_vs_branch_protection_doc": extra_vs_doc,
            "missing_vs_pr_template": missing_vs_pr,
            "extra_vs_pr_template": extra_vs_pr,
            "extra_report_preflight_vs_pr_template": extra_report_preflight_vs_pr,
            "extra_non_report_preflight_vs_pr_template": extra_non_report_preflight_vs_pr,
            "unexpected_optional_vs_pr_template": unexpected_optional_vs_pr,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def write_required_checks_snapshot(
    *,
    output_path: Path,
    report_workflow_path: Path,
    drift_workflow_path: Path,
    branch_protection_doc_path: Path,
    pr_template_path: Path,
) -> dict[str, Any]:
    snapshot = build_required_checks_snapshot(
        report_workflow_path=report_workflow_path,
        drift_workflow_path=drift_workflow_path,
        branch_protection_doc_path=branch_protection_doc_path,
        pr_template_path=pr_template_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def _report_preflight_job_names(path: Path) -> set[str]:
    payload = _read_yaml_mapping(path)
    jobs = payload.get("jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError(f"`jobs` must be a mapping in workflow: {path}")
    return {str(name) for name in jobs.keys()}


def _drift_required_checks(path: Path) -> set[str]:
    payload = _read_yaml_mapping(path)
    jobs = payload.get("jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError(f"`jobs` must be a mapping in workflow: {path}")
    job = jobs.get("verify_branch_protection", {})
    if not isinstance(job, dict):
        raise ValueError("workflow missing jobs.verify_branch_protection mapping")
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("jobs.verify_branch_protection.steps must be a list")

    run_blob = "\n".join(
        str(step.get("run", ""))
        for step in steps
        if isinstance(step, dict) and step.get("name") == "Verify branch protection required checks"
    )
    if not run_blob:
        raise ValueError("Unable to find run command for branch protection verification step")
    return set(re.findall(r'--required-check\s+"([^"]+)"', run_blob))


def _branch_protection_doc_required_checks(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("Required checks:")
    except ValueError as exc:
        raise ValueError(f"Required checks section missing in {path}") from exc

    contexts: set[str] = set()
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Optional "):
            break
        if not stripped.startswith("- "):
            continue
        for match in re.findall(r"`([^`]+)`", stripped):
            contexts.add(match)
    return contexts


def _pr_template_checklist_checks(path: Path) -> set[str]:
    contexts: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("- [ ]"):
            continue
        for match in re.findall(r"`([^`]+)`", line):
            if " / " in match:
                contexts.add(match)
    return contexts


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Write a required-checks snapshot JSON artifact")
    parser.add_argument("--output", default="artifacts/ci/required_checks_snapshot.json")
    parser.add_argument("--report-workflow", default=".github/workflows/report_preflight.yml")
    parser.add_argument("--drift-workflow", default=".github/workflows/branch_protection_drift.yml")
    parser.add_argument("--branch-protection-doc", default="docs/ci/branch_protection.md")
    parser.add_argument("--pr-template", default=".github/pull_request_template.md")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check mismatch is detected")
    parser.add_argument("--json", action="store_true", help="Print snapshot JSON to stdout")
    args = parser.parse_args(argv)

    snapshot = write_required_checks_snapshot(
        output_path=Path(args.output).resolve(),
        report_workflow_path=Path(args.report_workflow).resolve(),
        drift_workflow_path=Path(args.drift_workflow).resolve(),
        branch_protection_doc_path=Path(args.branch_protection_doc).resolve(),
        pr_template_path=Path(args.pr_template).resolve(),
    )
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    if args.strict and not bool(snapshot["pass"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
