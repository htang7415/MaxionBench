"""Archive benchmark run artifacts into a versioned, compressed bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DOCS: list[str] = ["command.md", "project.md", "prompt.md", "document.md"]

# Relative paths from repo root → label in the archive directory
_DEFAULT_ARTIFACT_DIRS: dict[str, str] = {
    "runs": "artifacts/runs/portable",
    "figures": "artifacts/figures/final",
    "hotpot_maxionbench": "dataset/processed/hotpot_portable",
    "conformance": "artifacts/conformance",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_item(src: Path, dest: Path, dry_run: bool) -> dict[str, Any]:
    if not src.exists():
        return {"src": str(src), "skipped": True, "reason": "not found"}
    if dry_run:
        kind = "dir" if src.is_dir() else "file"
        return {"src": str(src), "dest": str(dest), "type": kind, "dry_run": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
        file_count = sum(1 for p in dest.rglob("*") if p.is_file())
        return {"src": str(src), "dest": str(dest), "type": "dir", "files": file_count}
    shutil.copy2(src, dest)
    return {
        "src": str(src),
        "dest": str(dest),
        "type": "file",
        "sha256": _sha256_file(dest),
    }


def archive_run(
    *,
    run_id: str | None,
    results_dir: Path,
    docs: list[str],
    artifact_dirs: dict[str, str],
    tar: bool,
    dry_run: bool,
    repo_root: Path,
) -> dict[str, Any]:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    archive_dir = results_dir / run_id
    summary: dict[str, Any] = {
        "run_id": run_id,
        "archive_dir": str(archive_dir),
        "dry_run": dry_run,
        "items": [],
    }

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for doc_name in docs:
        item = _copy_item(repo_root / doc_name, archive_dir / doc_name, dry_run)
        item["label"] = f"doc:{doc_name}"
        summary["items"].append(item)

    for label, rel_path in artifact_dirs.items():
        item = _copy_item(repo_root / rel_path, archive_dir / label, dry_run)
        item["label"] = label
        summary["items"].append(item)

    if not dry_run:
        manifest_path = archive_dir / "archive_manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        summary["manifest"] = str(manifest_path)

    if tar:
        tar_path = results_dir / f"{run_id}.tar.gz"
        summary["tar_gz"] = str(tar_path)
        if not dry_run:
            shutil.make_archive(
                base_name=str(results_dir / run_id),
                format="gztar",
                root_dir=str(results_dir),
                base_dir=run_id,
            )
            if tar_path.exists():
                summary["tar_gz_bytes"] = tar_path.stat().st_size

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    run_id = summary["run_id"]
    archive_dir = summary["archive_dir"]
    dry_run = summary.get("dry_run", False)
    tag = "[dry-run] " if dry_run else ""

    print(f"{tag}run_id:  {run_id}")
    print(f"{tag}archive: {archive_dir}")

    for item in summary.get("items", []):
        label = item.get("label", "?")
        if item.get("skipped"):
            print(f"  {label}: skipped ({item.get('reason', 'not found')})")
        elif item.get("dry_run"):
            t = item.get("type", "?")
            print(f"  {label}: {item['src']} -> {item['dest']} ({t}, dry-run)")
        elif item.get("type") == "dir":
            print(f"  {label}: {item.get('files', 0)} files")
        else:
            sha = item.get("sha256", "")
            print(f"  {label}: {sha[:12]}..." if sha else f"  {label}: copied")

    tar_gz = summary.get("tar_gz")
    if tar_gz:
        size = summary.get("tar_gz_bytes")
        if size:
            size_mb = size / (1024 * 1024)
            print(f"  tar.gz:  {tar_gz} ({size_mb:.1f} MB)")
        else:
            print(f"  tar.gz:  {tar_gz}" + (" (dry-run)" if dry_run else ""))


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Archive benchmark run artifacts into a versioned bundle")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Archive identifier (default: UTC timestamp like 20260421T120000Z)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for archives (default: results/)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Override source for run artifacts (default: artifacts/runs/portable)",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Override source for figures (default: artifacts/figures/final)",
    )
    parser.add_argument(
        "--hotpot-maxionbench-dir",
        default=None,
        help="Override source for HotpotQA-MaxionBench processed data",
    )
    parser.add_argument("--hotpot-portable-dir", dest="hotpot_maxionbench_dir", default=None, help=SUPPRESS)
    parser.add_argument(
        "--conformance-dir",
        default=None,
        help="Override source for conformance outputs (default: artifacts/conformance)",
    )
    parser.add_argument(
        "--docs",
        default=None,
        help="Comma-separated doc filenames to include (default: command.md,project.md,prompt.md,document.md)",
    )
    parser.add_argument("--no-tar", action="store_true", help="Skip tar.gz creation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be archived without copying")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    docs = [d.strip() for d in args.docs.split(",")] if args.docs else _DEFAULT_DOCS

    artifact_dirs = dict(_DEFAULT_ARTIFACT_DIRS)
    if args.runs_dir:
        artifact_dirs["runs"] = args.runs_dir
    if args.figures_dir:
        artifact_dirs["figures"] = args.figures_dir
    if args.hotpot_maxionbench_dir:
        artifact_dirs["hotpot_maxionbench"] = args.hotpot_maxionbench_dir
    if args.conformance_dir:
        artifact_dirs["conformance"] = args.conformance_dir

    try:
        summary = archive_run(
            run_id=args.run_id,
            results_dir=Path(args.results_dir).resolve(),
            docs=docs,
            artifact_dirs=artifact_dirs,
            tar=not args.no_tar,
            dry_run=args.dry_run,
            repo_root=Path.cwd(),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
