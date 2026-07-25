"""Build clean package artifacts outside metadata-emulating filesystems."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT_FILES = ("LICENSE", "README.md", "pyproject.toml")


def _ignore_generated(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name.startswith("._") or name in {"__pycache__", ".pytest_cache"}
    }


def _verify_artifact(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    if any(Path(name).name.startswith("._") for name in names):
        raise RuntimeError(f"AppleDouble metadata found in package artifact: {path}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "dist"
    output_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="maxionbench-build-") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        for name in ROOT_FILES:
            shutil.copyfile(repo_root / name, stage / name)
        shutil.copytree(
            repo_root / "maxionbench",
            stage / "maxionbench",
            copy_function=shutil.copyfile,
            ignore=_ignore_generated,
        )

        env = dict(os.environ)
        env["COPYFILE_DISABLE"] = "1"
        subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", str(stage)],
            check=True,
            env=env,
        )

        artifacts = sorted((stage / "dist").iterdir())
        for artifact in artifacts:
            _verify_artifact(artifact)
            shutil.copyfile(artifact, output_dir / artifact.name)

    print("\n".join(str(output_dir / artifact.name) for artifact in artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
