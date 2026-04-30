"""Quick reviewer-facing artifact checks for the NeurIPS paper bundle."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "paper/artifact_card.md",
    "paper/README.md",
    "paper/archive/archive_manifest.json",
    "paper/manuscript/main.pdf",
    "paper/manuscript/main.tex",
    "paper/manuscript/tables/evidence_strength.tex",
    "paper/tables/neurips_main_results.csv",
    "paper/tables/portable_decision_table.csv",
    "paper/experiments/s2_mini_bundle/s2_b2_deadline_mini_bundle_summary.json",
    "paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json",
    "paper/experiments/s3_paired_quality/summary.json",
    "paper/metadata/hotpotqa_portable_croissant.jsonld",
    "paper/metadata/maxionbench_evaluation_card.json",
]


def _load_json(path: Path, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive reviewer script
        errors.append(f"invalid JSON: {path}: {exc}")
        return None


def _pdf_pages(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["pdfinfo", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def run_checks() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for rel_path in REQUIRED_FILES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            errors.append(f"missing required file: {rel_path}")

    sidecars = sorted(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "paper").rglob("._*"))
    if sidecars:
        errors.append(f"macOS sidecar files present under paper/: {sidecars[:10]}")

    archive = _load_json(REPO_ROOT / "paper/archive/archive_manifest.json", errors)
    if archive is not None:
        items = archive.get("items", [])
        labels = {item.get("label") for item in items if isinstance(item, dict)}
        for label in {"runs", "figures", "hotpot_portable", "conformance"}:
            if label not in labels:
                errors.append(f"archive manifest missing label: {label}")

    mini = _load_json(REPO_ROOT / "paper/experiments/s2_mini_bundle/s2_b2_deadline_mini_bundle_summary.json", errors)
    if mini is not None:
        validation = mini.get("validation", {})
        if validation.get("strict_schema_pass") is not True:
            errors.append("S2 mini-bundle summary does not report strict_schema_pass=true")
        if validation.get("error_count") != 0:
            errors.append("S2 mini-bundle summary reports nonzero validation errors")
        engines = set((mini.get("aggregate") or {}).keys())
        if engines != {"faiss-cpu", "qdrant"}:
            errors.append(f"S2 mini-bundle aggregate engines mismatch: {sorted(engines)}")
        paired = ((mini.get("paired_quality") or {}).get("ndcg_at_10") or {})
        if paired.get("paired_count") != 500:
            warnings.append("S2 mini-bundle paired nDCG count is not 500")
        fresh = ((mini.get("paired_freshness") or {}).get("freshness_hit_at_5s") or {})
        if fresh.get("mean_delta_qdrant_minus_faiss") != 0.0:
            errors.append("S2 mini-bundle freshness delta is not zero")

    larger_s2 = _load_json(
        REPO_ROOT / "paper/experiments/s2_larger_same_machine/s2_larger_same_machine_summary.json",
        errors,
    )
    if larger_s2 is not None:
        validation = larger_s2.get("validation", {})
        if larger_s2.get("status") != "completed":
            errors.append("S2 larger same-machine summary is not completed")
        if validation.get("error_count") != 0:
            errors.append("S2 larger same-machine summary reports validation errors")
        engines = set((larger_s2.get("aggregate") or {}).keys())
        if engines != {"faiss-cpu", "qdrant"}:
            errors.append(f"S2 larger same-machine aggregate engines mismatch: {sorted(engines)}")
        paired = ((larger_s2.get("paired_quality") or {}).get("ndcg_at_10") or {})
        if paired.get("paired_count") != 1788:
            warnings.append("S2 larger same-machine paired nDCG count is not 1788")
        fresh = ((larger_s2.get("paired_freshness") or {}).get("freshness_hit_at_5s") or {})
        if fresh.get("paired_count") != 200:
            warnings.append("S2 larger same-machine paired freshness count is not 200")
        if fresh.get("mean_delta_qdrant_minus_faiss") != 0.0:
            errors.append("S2 larger same-machine freshness delta is not zero")

    croissant = _load_json(REPO_ROOT / "paper/metadata/hotpotqa_portable_croissant.jsonld", errors)
    if croissant is not None:
        if croissant.get("conformsTo") != "http://mlcommons.org/croissant/1.0":
            warnings.append("Croissant metadata does not declare Croissant 1.0 conformance")
        if croissant.get("license") != "https://creativecommons.org/licenses/by-sa/4.0/":
            warnings.append("HotpotQA-portable license metadata should be checked before submission")
        if "ANONYMIZED_REVIEW_ARTIFACT_URL" in json.dumps(croissant):
            warnings.append("Croissant metadata still contains anonymized review URL placeholder")

    evaluation_card = _load_json(REPO_ROOT / "paper/metadata/maxionbench_evaluation_card.json", errors)
    if evaluation_card is not None and not evaluation_card.get("not_supported"):
        errors.append("evaluation card must list unsupported claims")

    pages = _pdf_pages(REPO_ROOT / "paper/manuscript/main.pdf")
    if pages is None:
        warnings.append("pdfinfo unavailable; page count was not checked")
    elif pages > 10:
        warnings.append(f"main.pdf is {pages} pages; verify submission-specific page accounting")

    return {
        "pass": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checked_files": len(REQUIRED_FILES),
        "pdf_pages": pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Verify NeurIPS paper artifact readiness files.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = run_checks()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        status = "PASS" if summary["pass"] else "FAIL"
        print(f"{status}: {summary['error_count']} errors, {summary['warning_count']} warnings")
        for error in summary["errors"]:
            print(f"error: {error}")
        for warning in summary["warnings"]:
            print(f"warning: {warning}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
