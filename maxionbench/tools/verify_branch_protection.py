"""Verify GitHub branch protection required checks for MaxionBench."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from typing import Any, Iterable, Mapping

import requests

DEFAULT_REQUIRED_CHECKS = (
    "report-preflight / conformance_readiness_gate",
    "report-preflight / report_preflight",
)
OPTIONAL_DRIFT_CHECK = "branch-protection-drift / verify_branch_protection"
OPTIONAL_REQUIRED_CHECKS = (OPTIONAL_DRIFT_CHECK,)


def extract_required_check_contexts(payload: Mapping[str, Any]) -> set[str]:
    required = payload.get("required_status_checks")
    if not isinstance(required, Mapping):
        return set()

    contexts: set[str] = set()
    raw_contexts = required.get("contexts")
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if isinstance(item, str) and item:
                contexts.add(item)

    raw_checks = required.get("checks")
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, Mapping):
                continue
            context = item.get("context")
            if isinstance(context, str) and context:
                contexts.add(context)
    return contexts


def evaluate_branch_protection(
    payload: Mapping[str, Any],
    *,
    required_checks: Iterable[str] = DEFAULT_REQUIRED_CHECKS,
) -> dict[str, Any]:
    required = sorted({str(item) for item in required_checks if str(item)})
    present = sorted(extract_required_check_contexts(payload))
    present_set = set(present)
    missing = [check for check in required if check not in present_set]
    return {
        "required_checks": required,
        "present_checks": present,
        "missing_checks": missing,
        "pass": not missing,
    }


def resolve_required_checks(
    required_checks: Iterable[str] | None,
    *,
    include_drift_check: bool,
    include_strict_readiness_check: bool = False,
    include_publish_bundle_check: bool = False,
) -> list[str]:
    del include_strict_readiness_check, include_publish_bundle_check
    checks = list(required_checks) if required_checks is not None else list(DEFAULT_REQUIRED_CHECKS)
    if include_drift_check:
        checks.append(OPTIONAL_DRIFT_CHECK)
    deduped: list[str] = []
    seen: set[str] = set()
    for check in checks:
        normalized = str(check)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def fetch_branch_protection(repo: str, branch: str, *, token: str | None, timeout_s: float) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/branches/{branch}/protection"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=timeout_s)
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed: {response.status_code} {response.reason} for {url}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub API response is not a JSON object")
    return payload


def verify_branch_protection(
    *,
    repo: str,
    branch: str,
    token: str | None,
    timeout_s: float,
    required_checks: Iterable[str] = DEFAULT_REQUIRED_CHECKS,
) -> dict[str, Any]:
    payload = fetch_branch_protection(repo, branch, token=token, timeout_s=timeout_s)
    result = evaluate_branch_protection(payload, required_checks=required_checks)
    result["repo"] = repo
    result["branch"] = branch
    return result


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify branch protection required checks")
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name format")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--token", default=None, help="GitHub token (defaults to env GITHUB_TOKEN)")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--required-check", action="append", dest="required_checks", default=None)
    parser.add_argument(
        "--include-drift-check",
        action="store_true",
        help=f"Also require `{OPTIONAL_DRIFT_CHECK}`",
    )
    parser.add_argument(
        "--include-strict-readiness-check",
        action="store_true",
        help="Deprecated no-op; strict-readiness workflow was removed from the portable track.",
    )
    parser.add_argument(
        "--include-publish-bundle-check",
        action="store_true",
        help="Deprecated no-op; publish-benchmark-bundle workflow was removed from the portable track.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("GITHUB_TOKEN")
    required_checks = resolve_required_checks(
        args.required_checks,
        include_drift_check=bool(args.include_drift_check),
        include_strict_readiness_check=bool(args.include_strict_readiness_check),
        include_publish_bundle_check=bool(args.include_publish_bundle_check),
    )
    summary = verify_branch_protection(
        repo=args.repo,
        branch=args.branch,
        token=token,
        timeout_s=args.timeout_s,
        required_checks=required_checks,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
