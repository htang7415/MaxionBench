"""Bootstrap a local/community dataset tree under ``dataset/``."""

from __future__ import annotations

from argparse import ArgumentParser
import bz2
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile

from maxionbench.tools.download_d1 import DEFAULT_HTTP_HEADERS, download_d1_dataset

ANN_HDF5_LAYOUT = {
    "D1/glove-100-angular.hdf5": "glove-100-angular",
    "D1/sift-128-euclidean.hdf5": "sift-128-euclidean",
    "D1/gist-960-euclidean.hdf5": "gist-960-euclidean",
    "D2/deep-image-96-angular.hdf5": "deep-image-96-angular",
}
BEIR_DATASETS = ("scifact", "fiqa", "nfcorpus")
BIGANN_SNAPSHOT_URL = "https://github.com/harsha-simhadri/big-ann-benchmarks/archive/refs/heads/main.tar.gz"
BIGANN_SNAPSHOT_ROOT_NAME = "big-ann-benchmarks-main"
CRAG_TASK12_URL = (
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/"
    "data/crag_task_1_and_2_dev_v4.jsonl.bz2?download="
)
_YFCC_DIR_CANDIDATES = ("yfcc100M", "yfcc-10M")
_BIGANN_RUNTIME_DISTRIBUTIONS = (
    "ansicolors",
    "docker",
    "h5py",
    "jinja2",
    "matplotlib",
    "numpy",
    "pandas",
    "psutil",
    "pyyaml",
    "scikit-learn",
    "scipy",
)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


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


def _locate_bigann_snapshot_root(extracted_root: Path) -> Path | None:
    direct = extracted_root / BIGANN_SNAPSHOT_ROOT_NAME
    if direct.is_dir():
        return direct
    candidates = sorted(
        path
        for path in extracted_root.iterdir()
        if path.is_dir() and path.name.startswith("big-ann-benchmarks-")
    )
    if candidates:
        return candidates[0]
    return None


def _bigann_snapshot_repo_is_usable(repo_dir: Path) -> bool:
    return (repo_dir / "create_dataset.py").is_file() and (repo_dir / "benchmark").is_dir()


def stage_bigann_snapshot(
    *,
    snapshot_url: str,
    repo_dir: Path,
    cache_dir: Path,
    timeout_s: float,
    force: bool,
) -> dict[str, str]:
    _validate_timeout(timeout_s)
    if repo_dir.exists() and not force:
        if _bigann_snapshot_repo_is_usable(repo_dir):
            return {"path": str(repo_dir.resolve()), "source": "cache_hit"}
        shutil.rmtree(repo_dir)

    archive_path = (cache_dir / f"{BIGANN_SNAPSHOT_ROOT_NAME}.tar.gz").resolve()
    archive_meta = download_file(url=snapshot_url, dest=archive_path, timeout_s=timeout_s, force=force)
    with tempfile.TemporaryDirectory(prefix="maxionbench_bigann_snapshot_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        extracted_root = _extract_tar_archive(archive_path=archive_path, workdir=tmpdir)
        src = _locate_bigann_snapshot_root(extracted_root)
        if src is None:
            raise FileNotFoundError(f"Unable to locate big-ann-benchmarks snapshot root in {archive_path}")
        copytree_replace(src=src, dst=repo_dir)
    return {"path": str(repo_dir.resolve()), "source": archive_meta["source"]}


def download_ann_benchmarks(*, root: Path, timeout_s: float, force: bool) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    fetched: dict[str, Any] = {}
    for rel_path, dataset_name in ANN_HDF5_LAYOUT.items():
        fetched[rel_path] = download_d1_dataset(
            dataset_name=dataset_name,
            output_path=(root / rel_path).resolve(),
            force=force,
            timeout_s=timeout_s,
        )
    return fetched


def _ensure_bigann_runtime_dependencies() -> None:
    missing: list[str] = []
    for dist_name in _BIGANN_RUNTIME_DISTRIBUTIONS:
        try:
            importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(dist_name)
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            "Missing big-ann runtime dependencies: "
            f"{joined}. Install MaxionBench with the datasets extra "
            '(`python -m pip install -e ".[datasets]"`) or rebuild the '
            "MaxionBench container image so D3 preparation runs without job-time pip installs."
        )


def download_bigann_yfcc(
    *,
    root: Path,
    cache_dir: Path,
    timeout_s: float,
    force: bool,
    requirements_file: str | None = None,
) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    dst = (root / "D3" / "yfcc-10M").resolve()
    if dst.exists() and not force:
        return {"path": str(dst), "source": "cache_hit"}

    repo_dir = (cache_dir / "big-ann-benchmarks").resolve()
    if not force:
        cached_src = _find_existing_yfcc_dir(repo_dir)
        if cached_src is not None:
            copytree_replace(src=cached_src, dst=dst)
            return {"path": str(dst), "source": "copied_from_snapshot_cache"}
    stage_bigann_snapshot(
        snapshot_url=BIGANN_SNAPSHOT_URL,
        repo_dir=repo_dir,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        force=force,
    )
    _ensure_bigann_runtime_dependencies()

    run([sys.executable, "create_dataset.py", "--dataset", "yfcc-10M"], cwd=repo_dir)
    src = _find_existing_yfcc_dir(repo_dir)
    if src is None:
        raise FileNotFoundError(
            "Could not locate YFCC dataset under big-ann-benchmarks/data "
            f"(tried {', '.join(_YFCC_DIR_CANDIDATES)})"
        )
    copytree_replace(src=src, dst=dst)
    return {"path": str(dst), "source": "copied_from_bigann_snapshot"}


def _find_existing_yfcc_dir(repo_dir: Path) -> Path | None:
    data_root = repo_dir / "data"
    if not data_root.exists():
        return None
    for name in _YFCC_DIR_CANDIDATES:
        src = data_root / name
        if src.exists():
            return src
    for name in _YFCC_DIR_CANDIDATES:
        found = find_dir_by_name(root=data_root, name=name)
        if found is not None:
            return found
    return None


def _extract_archive(*, archive_path: Path, workdir: Path) -> Path:
    extract_root = workdir / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)
    return extract_root


def download_beir(*, root: Path, timeout_s: float, force: bool) -> dict[str, Any]:
    _validate_timeout(timeout_s)
    beir_root = (root / "D4" / "beir").resolve()
    beir_root.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, Any] = {}
    for dataset in BEIR_DATASETS:
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


def write_manifest(*, root: Path, crag_examples: int) -> Path:
    payload = {
        "profile": "community_bootstrap",
        "note": (
            "Local/community dataset bootstrap. This tree is convenient for development, "
            "but it does not replace the pinned paper-lane D2/D3/D4 dataset requirements in prompt.md."
        ),
        "D1": [
            "glove-100-angular.hdf5",
            "sift-128-euclidean.hdf5",
            "gist-960-euclidean.hdf5",
        ],
        "D2": [
            "deep-image-96-angular.hdf5",
        ],
        "D3": [
            "yfcc-10M/",
        ],
        "D4": {
            "beir": list(BEIR_DATASETS),
            "crag_source_archive": "crag_task_1_and_2_dev_v4.jsonl.bz2",
            "crag_small_slice": f"crag_task_1_and_2_dev_v4.first_{crag_examples}.jsonl",
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
    skip_d1d2: bool = False,
    skip_d3: bool = False,
    skip_d4: bool = False,
    force: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(root),
        "cache_dir": str(cache_dir),
        "community_bootstrap": True,
        "warning": (
            "This downloader populates the requested local/community layout. "
            "Pinned paper-lane D2/D3/D4 datasets remain the source-of-truth in prompt.md/document.md."
        ),
        "fetched": {},
    }
    if not skip_d1d2:
        summary["fetched"]["d1_d2"] = download_ann_benchmarks(root=root, timeout_s=timeout_s, force=force)
    if not skip_d3:
        summary["fetched"]["d3"] = download_bigann_yfcc(
            root=root,
            cache_dir=cache_dir,
            timeout_s=timeout_s,
            force=force,
        )
    if not skip_d4:
        summary["fetched"]["d4_beir"] = download_beir(root=root, timeout_s=timeout_s, force=force)
        summary["fetched"]["d4_crag"] = download_crag(
            root=root,
            timeout_s=timeout_s,
            force=force,
            max_examples=crag_examples,
        )
    manifest_path = write_manifest(root=root, crag_examples=crag_examples)
    summary["manifest_path"] = str(manifest_path)
    return summary


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Download the requested local/community dataset tree under dataset/")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--crag-examples", type=int, default=500)
    parser.add_argument("--skip-d1d2", action="store_true")
    parser.add_argument("--skip-d3", action="store_true")
    parser.add_argument("--skip-d4", action="store_true")
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
        skip_d1d2=bool(args.skip_d1d2),
        skip_d3=bool(args.skip_d3),
        skip_d4=bool(args.skip_d4),
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
