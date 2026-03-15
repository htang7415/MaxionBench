"""Download D1 ann-benchmarks HDF5 bundles into the local repo cache."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

ANN_BENCHMARKS_BASE_URL = "https://ann-benchmarks.com"
DEFAULT_D1_OUTPUT_ROOT = Path("data") / "d1"
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def default_output_path(*, dataset_name: str, repo_root: Path | None = None) -> Path:
    _validate_dataset_name(dataset_name)
    base = Path.cwd() if repo_root is None else Path(repo_root).expanduser().resolve()
    return (base / DEFAULT_D1_OUTPUT_ROOT / f"{dataset_name}.hdf5").resolve()


def download_d1_dataset(
    *,
    dataset_name: str,
    output_path: Path | None = None,
    force: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, object]:
    _validate_dataset_name(dataset_name)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    target_path = (output_path.expanduser().resolve() if output_path is not None else default_output_path(dataset_name=dataset_name))
    url = f"{ANN_BENCHMARKS_BASE_URL}/{dataset_name}.hdf5"
    if target_path.exists() and not force:
        return {
            "dataset_name": dataset_name,
            "output_path": str(target_path),
            "url": url,
            "sha256": _sha256_file(target_path),
            "source": "cache_hit",
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f"{target_path.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            with urlopen(url, timeout=float(timeout_s)) as response:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise
    return {
        "dataset_name": dataset_name,
        "output_path": str(target_path),
        "url": url,
        "sha256": _sha256_file(target_path),
        "source": "download",
    }


def _validate_dataset_name(dataset_name: str) -> None:
    normalized = str(dataset_name).strip()
    if not normalized or not _DATASET_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "dataset_name must contain only letters, digits, dots, underscores, and hyphens "
            "and may not include path separators"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Download a D1 ann-benchmarks HDF5 bundle into the local data cache")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = download_d1_dataset(
        dataset_name=args.dataset_name,
        output_path=Path(args.output).expanduser() if args.output else None,
        force=bool(args.force),
        timeout_s=float(args.timeout_s),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(str(summary["output_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
