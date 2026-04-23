"""Bootstrap the portable-agentic dataset tree under ``dataset/``."""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile

DEFAULT_HTTP_HEADERS = {
    "User-Agent": "MaxionBench/0.1",
}

BEIR_DATASETS = ("scifact", "fiqa")
HOTPOTQA_DEV_DISTRACTOR_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
CRAG_TASK12_URL = (
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/"
    "data/crag_task_1_and_2_dev_v4.jsonl.bz2?download="
)


def _validate_timeout(timeout_s: float) -> None:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")


def download_file(*, url: str, dest: Path, timeout_s: float = 60.0, force: bool = False) -> dict[str, str]:
    _validate_timeout(timeout_s)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return {"url": url, "path": str(dest.resolve()), "source": "cache_hit"}

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest.parent,
            prefix=f"{dest.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            request = Request(url, headers=dict(DEFAULT_HTTP_HEADERS))
            with urlopen(request, timeout=float(timeout_s)) as response:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(dest)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise
    return {"url": url, "path": str(dest.resolve()), "source": "download"}


def copytree_replace(*, src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def find_dir_by_name(*, root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_dir():
            return path
    return None


def _extract_tar_archive(*, archive_path: Path, workdir: Path) -> Path:
    extract_root = workdir / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    root_path = str(extract_root.resolve())
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            dest_path = str((extract_root / member.name).resolve())
            if os.path.commonpath((root_path, dest_path)) != root_path:
                raise ValueError(f"Refusing to extract archive member outside destination: {member.name}")
        archive.extractall(extract_root)
    return extract_root


def _extract_archive(*, archive_path: Path, workdir: Path) -> Path:
    extract_root = workdir / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)
    return extract_root


def download_beir(*, root: Path, timeout_s: float, force: bool) -> dict[str, Any]:
    return download_beir_subsets(root=root, timeout_s=timeout_s, force=force, subsets=BEIR_DATASETS)


def download_beir_subsets(
    *,
    root: Path,
    timeout_s: float,
    force: bool,
    subsets: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    beir_root = (root / "D4" / "beir").resolve()
    beir_root.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, Any] = {}
    for dataset in subsets:
        dst = beir_root / dataset
        if dst.exists() and not force:
            fetched[dataset] = {"path": str(dst), "source": "cache_hit"}
            continue
        with tempfile.TemporaryDirectory(prefix=f"maxionbench_beir_{dataset}_") as tmpdir_raw:
            tmpdir = Path(tmpdir_raw)
            archive_path = tmpdir / f"{dataset}.zip"
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            download_file(url=url, dest=archive_path, timeout_s=timeout_s, force=True)
            extracted_root = _extract_archive(archive_path=archive_path, workdir=tmpdir)
            src = find_dir_by_name(root=extracted_root, name=dataset)
            if src is None:
                raise FileNotFoundError(f"Unable to locate BEIR dataset `{dataset}` in {archive_path}")
            copytree_replace(src=src, dst=dst)
        fetched[dataset] = {"path": str(dst), "source": "download"}
    return fetched


def make_small_crag_slice(*, source_bz2: Path, output_jsonl: Path, max_examples: int, force: bool) -> dict[str, Any]:
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if output_jsonl.exists() and not force:
        line_count = sum(1 for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
        return {"path": str(output_jsonl.resolve()), "examples": line_count, "source": "cache_hit"}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with bz2.open(source_bz2, "rt", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_examples:
                break
    return {"path": str(output_jsonl.resolve()), "examples": count, "source": "generated"}


def download_crag(*, root: Path, timeout_s: float, force: bool, max_examples: int) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    crag_root = (root / "D4" / "crag").resolve()
    crag_root.mkdir(parents=True, exist_ok=True)
    full_bz2 = crag_root / "crag_task_1_and_2_dev_v4.jsonl.bz2"
    full_meta = download_file(url=CRAG_TASK12_URL, dest=full_bz2, timeout_s=timeout_s, force=force)
    slice_path = crag_root / f"crag_task_1_and_2_dev_v4.first_{max_examples}.jsonl"
    slice_meta = make_small_crag_slice(
        source_bz2=full_bz2,
        output_jsonl=slice_path,
        max_examples=max_examples,
        force=force,
    )
    return {"archive": full_meta, "slice": slice_meta}


def download_hotpotqa(*, root: Path, timeout_s: float, force: bool) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    hotpot_root = (root / "D4" / "hotpotqa").resolve()
    hotpot_root.mkdir(parents=True, exist_ok=True)
    dest = hotpot_root / "hotpot_dev_distractor_v1.json"
    return download_file(
        url=HOTPOTQA_DEV_DISTRACTOR_URL,
        dest=dest,
        timeout_s=timeout_s,
        force=force,
    )


def write_manifest(*, root: Path, crag_examples: int, requested_datasets: list[str] | None = None) -> Path:
    payload = {
        "profile": "portable-agentic-bootstrap",
        "note": (
            "Portable-agentic dataset bootstrap for the Apple Silicon paper path. "
            "This manifest tracks the local text corpora and one-time manual workspaces required by project.md."
        ),
        "requested_datasets": list(requested_datasets or []),
        "D4": {
            "beir": list(BEIR_DATASETS),
            "crag_source_archive": "crag_task_1_and_2_dev_v4.jsonl.bz2",
            "crag_small_slice": f"crag_task_1_and_2_dev_v4.first_{crag_examples}.jsonl",
            "hotpotqa_dev_distractor": "D4/hotpotqa/hotpot_dev_distractor_v1.json",
        },
    }
    path = (root / "manifest.json").resolve()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def download_datasets(
    *,
    root: Path,
    cache_dir: Path,
    crag_examples: int,
    datasets: list[str] | None = None,
    force: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip().lower() for item in (datasets or []) if item.strip()}
    selected_beir = [dataset for dataset in BEIR_DATASETS if dataset in selected]
    want_hotpotqa = "hotpotqa" in selected
    want_d4 = True if not selected else bool(selected_beir or selected & {"d4", "crag", "hotpotqa"})

    summary: dict[str, Any] = {
        "root": str(root),
        "cache_dir": str(cache_dir),
        "requested_datasets": sorted(selected),
        "portable_agentic_bootstrap": True,
        "warning": (
            "This downloader populates the portable-agentic text datasets and portable S3 source files only. "
            "The source-of-truth scope is project.md."
        ),
        "fetched": {},
    }
    if want_d4:
        if not selected or selected_beir:
            summary["fetched"]["d4_beir"] = download_beir_subsets(
                root=root,
                timeout_s=timeout_s,
                force=force,
                subsets=tuple(selected_beir or BEIR_DATASETS),
            )
        if not selected or "crag" in selected:
            summary["fetched"]["d4_crag"] = download_crag(
                root=root,
                timeout_s=timeout_s,
                force=force,
                max_examples=crag_examples,
            )
        if not selected or want_hotpotqa:
            summary["fetched"]["d4_hotpotqa"] = download_hotpotqa(
                root=root,
                timeout_s=timeout_s,
                force=force,
            )
    manifest_path = write_manifest(root=root, crag_examples=crag_examples, requested_datasets=sorted(selected))
    summary["manifest_path"] = str(manifest_path)
    return summary


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Download the portable-agentic dataset tree under dataset/")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset selection")
    parser.add_argument("--crag-examples", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = download_datasets(
        root=args.root,
        cache_dir=args.cache_dir,
        crag_examples=int(args.crag_examples),
        datasets=[item for item in str(args.datasets).split(",") if item.strip()],
        force=bool(args.force),
        timeout_s=float(args.timeout_s),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary["root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
