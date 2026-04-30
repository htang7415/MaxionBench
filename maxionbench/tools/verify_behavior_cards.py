"""Verify engine behavior-card coverage and required semantic sections."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

REQUIRED_CARD_FILES = (
    "faiss_cpu.md",
    "lancedb.md",
    "pgvector.md",
    "qdrant.md",
)
FORBIDDEN_LANCEDB_SPLIT_FILES = (
    "lancedb_service.md",
    "lancedb-inproc.md",
    "lancedb_inproc.md",
)
REQUIRED_SECTION_KEYWORDS = (
    "## engine",
    "## visibility semantics",
    "## delete semantics",
    "## update semantics",
    "## compaction",
    "## persistence",
)
REQUIRED_LIMITATIONS_KEYWORDS = (
    "## limitations",
    "## unsupported features",
)


def verify_behavior_cards(behavior_dir: Path) -> dict[str, Any]:
    root = behavior_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Behavior directory does not exist: {root}")

    errors: list[dict[str, Any]] = []
    files = sorted(path.name for path in root.glob("*.md"))
    file_set = set(files)

    for filename in REQUIRED_CARD_FILES:
        if filename not in file_set:
            errors.append(
                {
                    "file": str(root / filename),
                    "field": "presence",
                    "expected": "exists",
                    "actual": "missing",
                    "message": f"required behavior card missing: {filename}",
                }
            )

    for forbidden in FORBIDDEN_LANCEDB_SPLIT_FILES:
        if forbidden in file_set:
            errors.append(
                {
                    "file": str(root / forbidden),
                    "field": "lancedb_policy",
                    "expected": "single shared lancedb.md card",
                    "actual": "split mode-specific card present",
                    "message": f"LanceDB policy violation: remove {forbidden} and keep shared lancedb.md sections",
                }
            )

    cards_checked = 0
    for filename in REQUIRED_CARD_FILES:
        path = root / filename
        if not path.exists():
            continue
        cards_checked += 1
        text = path.read_text(encoding="utf-8")
        errors.extend(_validate_card(path=path, text=text))
        if filename == "lancedb.md":
            errors.extend(_validate_lancedb_shared_card(path=path, text=text))

    return {
        "behavior_dir": str(root),
        "files_checked": cards_checked,
        "error_count": len(errors),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _validate_card(*, path: Path, text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    errors: list[dict[str, Any]] = []
    for keyword in REQUIRED_SECTION_KEYWORDS:
        if keyword in lowered:
            continue
        errors.append(
            {
                "file": str(path),
                "field": "section",
                "expected": keyword,
                "actual": "missing",
                "message": f"{path.name} missing required section keyword: {keyword}",
            }
        )
    if not any(keyword in lowered for keyword in REQUIRED_LIMITATIONS_KEYWORDS):
        errors.append(
            {
                "file": str(path),
                "field": "section",
                "expected": "limitations/unsupported features section",
                "actual": "missing",
                "message": f"{path.name} missing required limitations section",
            }
        )
    return errors


def _validate_lancedb_shared_card(*, path: Path, text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    errors: list[dict[str, Any]] = []
    if "mode: lancedb-service" not in lowered:
        errors.append(
            {
                "file": str(path),
                "field": "lancedb_policy",
                "expected": "Mode: lancedb-service section",
                "actual": "missing",
                "message": "lancedb.md must include explicit section for lancedb-service",
            }
        )
    if "mode: lancedb-inproc" not in lowered:
        errors.append(
            {
                "file": str(path),
                "field": "lancedb_policy",
                "expected": "Mode: lancedb-inproc section",
                "actual": "missing",
                "message": "lancedb.md must include explicit section for lancedb-inproc",
            }
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify behavior-card coverage and required semantic sections")
    parser.add_argument("--behavior-dir", default="docs/behavior")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_behavior_cards(Path(args.behavior_dir))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if summary["pass"]:
            print(f"behavior-card verification passed: {summary['files_checked']} files checked")
        else:
            print(f"behavior-card verification failed: {summary['error_count']} issue(s)")
            for item in summary["errors"]:
                print(f"- {item['message']}")
    return 0 if bool(summary["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
