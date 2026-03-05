"""Inspect report metadata sidecars for output-policy provenance consistency."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import re
from typing import Any, Mapping

ALLOWED_MODES = {"milestones", "final"}
ALLOWED_OUTPUT_PATH_CLASSES = {"final", "milestones_noncanonical", "milestones_mx"}
MILESTONE_ID_RE = re.compile(r"^M[0-9]+$")


def inspect_report_output_policy(input_dir: Path) -> dict[str, Any]:
    root = input_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    expected_resolved_out_dir = str(root)
    expected_milestone_root = str(Path("artifacts/figures/milestones").resolve())

    meta_paths = sorted(root.rglob("*.meta.json"))
    no_meta_files = len(meta_paths) == 0
    invalid_json_files: list[str] = []
    missing_output_policy_files: list[str] = []
    invalid_output_policy: list[dict[str, Any]] = []
    output_path_class_counts: dict[str, int] = {}
    modes: set[str] = set()
    milestone_ids: set[str] = set()
    resolved_out_dirs: set[str] = set()
    milestone_roots: set[str] = set()

    for path in meta_paths:
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            invalid_json_files.append(str(path))
            continue
        if not isinstance(payload, Mapping):
            invalid_output_policy.append({"path": str(path), "issues": ["meta payload must be a JSON object"]})
            continue
        policy = payload.get("output_policy")
        if not isinstance(policy, Mapping):
            missing_output_policy_files.append(str(path))
            continue
        issues = _validate_output_policy(policy)
        if issues:
            invalid_output_policy.append({"path": str(path), "issues": issues})
            continue

        mode = str(policy["mode"])
        cls = str(policy["output_path_class"])
        modes.add(mode)
        output_path_class_counts[cls] = output_path_class_counts.get(cls, 0) + 1
        resolved_out_dirs.add(str(policy["resolved_out_dir"]))
        milestone_id = policy.get("milestone_id")
        if isinstance(milestone_id, str) and milestone_id:
            milestone_ids.add(milestone_id)
        milestone_root = policy.get("milestone_root")
        if isinstance(milestone_root, str) and milestone_root:
            milestone_roots.add(milestone_root)

    mixed_modes = len(modes) > 1
    mixed_output_path_classes = len(output_path_class_counts) > 1
    mixed_milestone_ids = len(milestone_ids) > 1
    mixed_resolved_out_dirs = len(resolved_out_dirs) > 1
    mixed_milestone_roots = len(milestone_roots) > 1
    expected_output_path_class, expected_milestone_id = _expected_output_policy_for_input(root=root, modes=modes)
    output_path_class_mismatch = (
        expected_output_path_class is not None
        and bool(output_path_class_counts)
        and set(output_path_class_counts.keys()) != {expected_output_path_class}
    )
    if expected_output_path_class == "milestones_mx":
        milestone_id_mismatch = set(milestone_ids) != {str(expected_milestone_id)}
    else:
        milestone_id_mismatch = bool(milestone_ids)
    resolved_out_dir_mismatch = (
        len(resolved_out_dirs) == 1 and next(iter(resolved_out_dirs)) != expected_resolved_out_dir
    )
    milestone_root_mismatch = len(milestone_roots) == 1 and next(iter(milestone_roots)) != expected_milestone_root
    error_count = (
        (1 if no_meta_files else 0)
        + (1 if mixed_modes else 0)
        + (1 if mixed_output_path_classes else 0)
        + (1 if mixed_milestone_ids else 0)
        + (1 if mixed_resolved_out_dirs else 0)
        + (1 if mixed_milestone_roots else 0)
        + (1 if output_path_class_mismatch else 0)
        + (1 if milestone_id_mismatch else 0)
        + (1 if resolved_out_dir_mismatch else 0)
        + (1 if milestone_root_mismatch else 0)
        + len(invalid_json_files)
        + len(missing_output_policy_files)
        + len(invalid_output_policy)
    )
    passed = (
        not no_meta_files
        and not mixed_modes
        and not mixed_output_path_classes
        and not mixed_milestone_ids
        and not mixed_resolved_out_dirs
        and not mixed_milestone_roots
        and not output_path_class_mismatch
        and not milestone_id_mismatch
        and not resolved_out_dir_mismatch
        and not milestone_root_mismatch
        and not invalid_json_files
        and not missing_output_policy_files
        and not invalid_output_policy
    )
    return {
        "input_dir": str(root),
        "meta_file_count": len(meta_paths),
        "no_meta_files": no_meta_files,
        "meta_files": [str(path) for path in meta_paths],
        "expected_output_path_class": expected_output_path_class,
        "expected_milestone_id": expected_milestone_id,
        "output_path_class_counts": dict(sorted(output_path_class_counts.items())),
        "modes": sorted(modes),
        "milestone_ids": sorted(milestone_ids),
        "expected_resolved_out_dir": expected_resolved_out_dir,
        "resolved_out_dirs": sorted(resolved_out_dirs),
        "expected_milestone_root": expected_milestone_root,
        "milestone_roots": sorted(milestone_roots),
        "mixed_modes": mixed_modes,
        "mixed_output_path_classes": mixed_output_path_classes,
        "mixed_milestone_ids": mixed_milestone_ids,
        "mixed_resolved_out_dirs": mixed_resolved_out_dirs,
        "mixed_milestone_roots": mixed_milestone_roots,
        "output_path_class_mismatch": output_path_class_mismatch,
        "milestone_id_mismatch": milestone_id_mismatch,
        "resolved_out_dir_mismatch": resolved_out_dir_mismatch,
        "milestone_root_mismatch": milestone_root_mismatch,
        "invalid_json_files": sorted(invalid_json_files),
        "missing_output_policy_files": sorted(missing_output_policy_files),
        "invalid_output_policy": invalid_output_policy,
        "error_count": error_count,
        "pass": passed,
    }


def _validate_output_policy(policy: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    mode = policy.get("mode")
    if mode not in ALLOWED_MODES:
        issues.append("`output_policy.mode` must be one of: milestones, final")
    resolved_out_dir = policy.get("resolved_out_dir")
    if not isinstance(resolved_out_dir, str) or not resolved_out_dir.strip():
        issues.append("`output_policy.resolved_out_dir` must be a non-empty string")

    cls = policy.get("output_path_class")
    if cls not in ALLOWED_OUTPUT_PATH_CLASSES:
        issues.append("`output_policy.output_path_class` must be one of: final, milestones_noncanonical, milestones_mx")
        return issues

    milestone_id = policy.get("milestone_id")
    if cls == "milestones_mx":
        if not isinstance(milestone_id, str) or not MILESTONE_ID_RE.fullmatch(milestone_id):
            issues.append("`output_policy.milestone_id` must match `M<integer>` for milestones_mx")
        milestone_root = policy.get("milestone_root")
        if not isinstance(milestone_root, str) or not milestone_root.strip():
            issues.append("`output_policy.milestone_root` must be a non-empty string for milestone classes")
    elif cls == "milestones_noncanonical":
        if milestone_id is not None:
            issues.append("`output_policy.milestone_id` must be null for milestones_noncanonical")
        milestone_root = policy.get("milestone_root")
        if not isinstance(milestone_root, str) or not milestone_root.strip():
            issues.append("`output_policy.milestone_root` must be a non-empty string for milestone classes")
    else:
        if milestone_id is not None:
            issues.append("`output_policy.milestone_id` must be null for final class")

    if isinstance(mode, str):
        if cls == "final" and mode != "final":
            issues.append("`output_policy.mode` must be `final` when output_path_class is `final`")
        if cls.startswith("milestones_") and mode != "milestones":
            issues.append("`output_policy.mode` must be `milestones` for milestone output classes")
    return issues


def _expected_output_policy_for_input(*, root: Path, modes: set[str]) -> tuple[str | None, str | None]:
    if len(modes) != 1:
        return None, None
    mode = next(iter(modes))
    if mode == "final":
        return "final", None

    milestone_root = Path("artifacts/figures/milestones").resolve()
    try:
        rel = root.relative_to(milestone_root)
    except ValueError:
        return "milestones_noncanonical", None
    if rel.parts and MILESTONE_ID_RE.fullmatch(rel.parts[0]):
        return "milestones_mx", rel.parts[0]
    return "milestones_noncanonical", None


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Inspect report metadata sidecars for output-policy consistency")
    parser.add_argument("--input", required=True, help="Report output directory containing *.meta.json sidecars")
    parser.add_argument("--output", default=None, help="Optional path to write JSON summary artifact")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when policy checks fail")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args(argv)

    summary = inspect_report_output_policy(Path(args.input))
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        status = "pass" if summary["pass"] else "fail"
        print(
            f"{status}: {summary['meta_file_count']} meta files checked, "
            f"{summary['error_count']} issue(s)"
        )
    if args.strict and not bool(summary["pass"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
