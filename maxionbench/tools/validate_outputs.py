"""Validate portable MaxionBench run artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_RUN_FILES = ("results.parquet", "run_metadata.json", "config_resolved.yaml")
PORTABLE_SCENARIOS = {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}


def validate_path(path: Path, *, strict_schema: bool = True, enforce_protocol: bool = False) -> dict[str, Any]:
    del strict_schema, enforce_protocol
    root = path.expanduser().resolve()
    run_dirs = _discover_run_dirs(root)
    errors: list[dict[str, Any]] = []
    checked: list[str] = []
    for run_dir in run_dirs:
        checked.append(str(run_dir))
        errors.extend(_validate_run_dir(run_dir))
    if root.exists() and not run_dirs:
        errors.append({"path": str(root), "message": "no run directories with results.parquet found"})
    if not root.exists():
        errors.append({"path": str(root), "message": "input path does not exist"})
    return {
        "input": str(root),
        "run_dirs_checked": len(checked),
        "run_dirs": checked,
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root.parent] if root.name == "results.parquet" else []
    if (root / "results.parquet").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("results.parquet")})


def _validate_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for name in REQUIRED_RUN_FILES:
        if not (run_dir / name).exists():
            errors.append({"path": str(run_dir / name), "message": f"missing required artifact {name}"})
    metadata = _read_json(run_dir / "run_metadata.json", errors)
    if metadata:
        scenario = str(metadata.get("scenario") or "")
        profile = str(metadata.get("profile") or "")
        if scenario not in PORTABLE_SCENARIOS:
            errors.append({"path": str(run_dir), "message": f"unsupported portable scenario {scenario!r}"})
        if profile not in {"maxionbench", "portable-agentic"}:
            errors.append({"path": str(run_dir), "message": f"metadata profile must be maxionbench, got {profile!r}"})
    try:
        frame = pd.read_parquet(run_dir / "results.parquet")
    except Exception as exc:
        errors.append({"path": str(run_dir / "results.parquet"), "message": f"failed to read results parquet: {exc}"})
        return errors
    required_columns = {"run_id", "scenario", "engine", "dataset_bundle", "search_params_json", "p99_ms", "qps"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        errors.append({"path": str(run_dir / "results.parquet"), "message": f"missing result columns: {missing}"})
    if "scenario" in frame.columns:
        bad = sorted({str(value) for value in frame["scenario"].dropna().tolist()} - PORTABLE_SCENARIOS)
        if bad:
            errors.append({"path": str(run_dir / "results.parquet"), "message": f"non-portable scenarios present: {bad}"})
    return errors


def _read_json(path: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append({"path": str(path), "message": f"invalid JSON: {exc}"})
        return {}
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "message": "JSON payload must be an object"})
        return {}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate portable MaxionBench output artifacts.")
    parser.add_argument("--input", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict-schema", action="store_true")
    mode.add_argument("--legacy-ok", action="store_true")
    parser.add_argument("--enforce-protocol", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = validate_path(
        Path(args.input),
        strict_schema=not bool(args.legacy_ok),
        enforce_protocol=bool(args.enforce_protocol),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif not summary["pass"]:
        for error in summary["errors"]:
            print(error["message"])
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
