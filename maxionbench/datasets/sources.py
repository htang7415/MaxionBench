"""Pinned v0.3 dataset sources: download, derive small slices, and verify SHA-256 under ``dataset/v03/``.

Every file a loader reads is listed in ``manifests/v03.yaml`` with its SHA-256: ``files`` are
downloaded as-is; ``derived`` files are built deterministically from a downloaded file so loaders
never parse multi-GB raw dumps. ``dataset/`` is git-ignored, so only the manifest is committed.

    python -m maxionbench.datasets.sources fetch [--group bfcl ...]
    python -m maxionbench.datasets.sources verify
"""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
import functools
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterator

import yaml

from maxionbench.datasets.cache_integrity import sha256_file, verify_file_sha256
from maxionbench.tools.download_datasets import download_file

DATASET_ROOT = Path("dataset/v03")
MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "v03.yaml"
MANIFEST_SCHEMA = "maxionbench-datasets-v03"


@functools.cache
def manifest() -> dict[str, Any]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"{MANIFEST_PATH}: schema_version must be {MANIFEST_SCHEMA!r}")
    return payload


def entry(rel: str) -> dict[str, Any]:
    m = manifest()
    found = m["files"].get(rel) or m["derived"].get(rel)
    if found is None:
        raise KeyError(f"{rel!r} is not in {MANIFEST_PATH.name}")
    return found


@functools.cache
def verified_path(rel: str, root: Path = DATASET_ROOT) -> Path:
    """Path of a manifest file after checking its SHA-256.

    A verified hash is remembered in `<root>/.sha256-cache.json` keyed by size and mtime, so loaders
    do not re-hash GB-sized files every run; `verify` always re-hashes.
    """
    path = Path(root) / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m maxionbench.datasets.sources fetch`")
    expected = entry(rel)["sha256"]
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns, expected]
    cache_path = Path(root) / ".sha256-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get(rel) != stamp:
        verify_file_sha256(path=path, expected_sha256=expected, label=rel)
        cache[rel] = stamp
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return path


def derive_crag_slice(src: Path, dst: Path, examples: int) -> None:
    """First `examples` CRAG records with the raw page HTML dropped (snippets and metadata kept)."""
    with bz2.open(src, "rt", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for n, line in enumerate(fin):
            if n >= examples:
                break
            row = json.loads(line)
            row["search_results"] = [
                {k: v for k, v in page.items() if k != "page_result"} for page in row.get("search_results", [])
            ]
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def derive_sharegpt_sample(src: Path, dst: Path, examples: int, seed: int = 0) -> None:
    """Seeded sample of first-turn (human prompt, assistant reply) pairs, as in vLLM's serving benchmark."""
    pairs = [
        {"id": row["id"], "prompt": conv[0]["value"], "completion": conv[1]["value"]}
        for row in _iter_json_array(src)
        if len(conv := row.get("conversations") or []) >= 2
        and conv[0].get("from") == "human" and conv[1].get("from") == "gpt"
        and conv[0]["value"].strip() and conv[1]["value"].strip()
    ]
    with dst.open("w", encoding="utf-8") as fout:
        for row in random.Random(seed).sample(pairs, examples):
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _iter_json_array(path: Path, chunk_chars: int = 1 << 20) -> Iterator[Any]:
    """Yield the items of a top-level JSON array without loading the whole file (ShareGPT is 673 MB)."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as fh:
        buf = fh.read(chunk_chars).lstrip()
        if not buf.startswith("["):
            raise ValueError(f"{path}: expected a JSON array")
        buf, pos, eof = buf[1:], 0, False
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = fh.read(chunk_chars)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            yield item
            pos = end


DERIVERS: dict[str, Callable[..., None]] = {"crag_slice": derive_crag_slice, "sharegpt_sample": derive_sharegpt_sample}


def fetch(groups: set[str] | None = None, root: Path = DATASET_ROOT, log: Callable[[str], None] = print) -> list[str]:
    """Download and derive every file in `groups` (all by default); returns checksum errors."""
    m = manifest()
    errors = []
    for rel, meta in m["files"].items():
        if groups and meta["group"] not in groups:
            continue
        dest = Path(root) / rel
        if not dest.exists() or dest.stat().st_size != int(meta["bytes"]):
            log(f"download {rel} ({int(meta['bytes']) / 1e6:.0f} MB)")
            download_file(url=meta["url"], dest=dest, timeout_s=600.0, force=True)
        errors += _check(dest, rel, meta, log)
    for rel, meta in m["derived"].items():
        if groups and meta["group"] not in groups:
            continue
        dest = Path(root) / rel
        if not dest.exists():
            log(f"derive {rel} from {meta['from']}")
            params = {k: v for k, v in meta.items() if k in ("examples", "seed")}
            DERIVERS[meta["kind"]](Path(root) / meta["from"], dest, **params)
        errors += _check(dest, rel, meta, log)
    return errors


def verify(root: Path = DATASET_ROOT, log: Callable[[str], None] = print) -> list[str]:
    m = manifest()
    errors = []
    for rel, meta in {**m["files"], **m["derived"]}.items():
        errors += _check(Path(root) / rel, rel, meta, log)
    return errors


def _check(path: Path, rel: str, meta: dict[str, Any], log: Callable[[str], None]) -> list[str]:
    if not path.exists():
        log(f"MISSING  {rel}")
        return [f"{rel}: missing"]
    actual = sha256_file(path)
    if actual != meta["sha256"]:
        log(f"MISMATCH {rel}: {actual}")
        return [f"{rel}: sha256 {actual} != {meta['sha256']}"]
    log(f"ok       {rel}")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--group", action="append", help="limit fetch to these groups (repeatable)")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        errors = fetch(set(args.group or []), args.root)
    else:
        errors = verify(args.root)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
