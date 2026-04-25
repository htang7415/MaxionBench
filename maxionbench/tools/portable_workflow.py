"""High-level workflow commands for the portable lane."""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator

from maxionbench.conformance.matrix import main as conformance_matrix_main
from maxionbench.reports.portable_exports import generate_portable_report_bundle
from maxionbench.tools.archive import main as archive_main
from maxionbench.tools.download_datasets import main as download_datasets_main
from maxionbench.tools.precompute_text_embeddings import main as precompute_text_embeddings_main
from maxionbench.tools.preprocess_datasets import main as preprocess_datasets_main
from maxionbench.tools.preprocess_hotpot_portable import main as preprocess_hotpot_portable_main
from maxionbench.tools.service_lifecycle import main as services_main


_DEFAULT_LANCEDB_INPROC_URI = "artifacts/lancedb/inproc"
_DEFAULT_HOTPOTQA_INPUT = "dataset/D4/hotpotqa/hotpot_dev_distractor_v1.json"
_DEFAULT_HOTPOT_PORTABLE_OUT = "dataset/processed/hotpot_portable"
_DEFAULT_D4_INPUT = "dataset/processed/D4"
_DEFAULT_RUNS_DIR = "artifacts/runs/portable"
_DEFAULT_FIGURES_DIR = "artifacts/figures/final"
_EMBEDDING_MODELS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5")
_DEFAULT_BEIR_SUBSETS = ("scifact", "fiqa")
_DEFAULT_CRAG_SLICE = "dataset/D4/crag/crag_task_1_and_2_dev_v4.first_500.jsonl"


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def ensure_lancedb_inproc_uri(*, repo_root: Path) -> str:
    current = str(os.environ.get("MAXIONBENCH_LANCEDB_INPROC_URI") or "").strip()
    if current:
        return current
    default_uri = str((repo_root / _DEFAULT_LANCEDB_INPROC_URI).resolve())
    os.environ["MAXIONBENCH_LANCEDB_INPROC_URI"] = default_uri
    return default_uri


def _require_success(step: str, exit_code: int | None) -> None:
    if exit_code not in (None, 0):
        raise RuntimeError(f"{step} failed with exit code {exit_code}")


def _preprocess_portable_d4() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for subset in _DEFAULT_BEIR_SUBSETS:
        input_dir = Path("dataset") / "D4" / "beir" / subset
        if not input_dir.exists():
            continue
        output_dir = Path(_DEFAULT_D4_INPUT) / "beir" / subset
        preprocess_datasets_main(
            [
                "beir",
                "--input",
                str(input_dir),
                "--out",
                str(output_dir),
                "--name",
                subset,
            ]
        )
        jobs.append({"kind": "beir", "input": str(input_dir), "output": str(output_dir)})

    crag_input = Path(_DEFAULT_CRAG_SLICE)
    if crag_input.exists():
        crag_output = Path(_DEFAULT_D4_INPUT) / "crag" / "small_slice"
        preprocess_datasets_main(
            [
                "crag",
                "--input",
                str(crag_input),
                "--out",
                str(crag_output),
            ]
        )
        jobs.append({"kind": "crag", "input": str(crag_input), "output": str(crag_output)})
    return jobs


def portable_setup(*, repo_root: Path) -> dict[str, Any]:
    resolved_repo_root = repo_root.expanduser().resolve()
    lancedb_uri = ensure_lancedb_inproc_uri(repo_root=resolved_repo_root)
    with _pushd(resolved_repo_root):
        _require_success(
            "portable services startup",
            services_main(["up", "--profile", "portable", "--wait", "--json"]),
        )
        _require_success(
            "portable conformance matrix",
            conformance_matrix_main(
                [
                    "--config-dir",
                    "configs/conformance",
                    "--out-dir",
                    "artifacts/conformance",
                    "--timeout-s",
                    "30",
                    "--adapters",
                    "mock,faiss-cpu,lancedb-inproc,qdrant,pgvector",
                ]
            ),
        )
    return {
        "mode": "portable-setup",
        "repo_root": str(resolved_repo_root),
        "lancedb_inproc_uri": lancedb_uri,
        "services_profile": "portable",
        "conformance_out_dir": str((resolved_repo_root / "artifacts/conformance").resolve()),
    }


def portable_data(*, repo_root: Path) -> dict[str, Any]:
    resolved_repo_root = repo_root.expanduser().resolve()
    ensure_lancedb_inproc_uri(repo_root=resolved_repo_root)
    with _pushd(resolved_repo_root):
        _require_success(
            "portable dataset download",
            download_datasets_main(
                [
                    "--root",
                    "dataset",
                    "--cache-dir",
                    ".cache",
                    "--datasets",
                    "scifact,fiqa,crag,hotpotqa",
                ]
            ),
        )
        d4_jobs = _preprocess_portable_d4()
        hotpot_input = (resolved_repo_root / _DEFAULT_HOTPOTQA_INPUT).resolve()
        if not hotpot_input.exists():
            raise FileNotFoundError(f"portable HotpotQA source missing: {hotpot_input}")
        hotpot_portable: dict[str, Any]
        _require_success(
            "portable HotpotQA preprocessing",
            preprocess_hotpot_portable_main(
                [
                    "--input",
                    _DEFAULT_HOTPOTQA_INPUT,
                    "--out",
                    _DEFAULT_HOTPOT_PORTABLE_OUT,
                ]
            ),
        )
        hotpot_portable = {
            "status": "processed",
            "input_path": str(hotpot_input),
            "output_dir": str((resolved_repo_root / _DEFAULT_HOTPOT_PORTABLE_OUT).resolve()),
        }
        embedding_jobs: list[dict[str, str]] = []
        embedding_inputs = [_DEFAULT_D4_INPUT]
        if hotpot_portable["status"] == "processed":
            embedding_inputs.append(_DEFAULT_HOTPOT_PORTABLE_OUT)
        for input_path in embedding_inputs:
            for model_id in _EMBEDDING_MODELS:
                _require_success(
                    f"portable embedding precompute for {input_path} with {model_id}",
                    precompute_text_embeddings_main(["--input", input_path, "--model-id", model_id]),
                )
                embedding_jobs.append({"input": input_path, "model_id": model_id})
    return {
        "mode": "portable-data",
        "repo_root": str(resolved_repo_root),
        "datasets": "scifact,fiqa,crag,hotpotqa",
        "d4_processed_root": str((resolved_repo_root / _DEFAULT_D4_INPUT).resolve()),
        "d4_preprocess_jobs": d4_jobs,
        "hotpot_portable": hotpot_portable,
        "hotpot_portable_out": str((resolved_repo_root / _DEFAULT_HOTPOT_PORTABLE_OUT).resolve()),
        "embedding_jobs": embedding_jobs,
    }


def portable_finalize(*, repo_root: Path) -> dict[str, Any]:
    resolved_repo_root = repo_root.expanduser().resolve()
    ensure_lancedb_inproc_uri(repo_root=resolved_repo_root)
    runs_dir = (resolved_repo_root / _DEFAULT_RUNS_DIR).resolve()
    figures_dir = (resolved_repo_root / _DEFAULT_FIGURES_DIR).resolve()
    with _pushd(resolved_repo_root):
        generate_portable_report_bundle(
            input_dir=runs_dir,
            out_dir=figures_dir,
            conformance_matrix_path=(resolved_repo_root / "artifacts" / "conformance" / "conformance_matrix.csv").resolve(),
            behavior_dir=(resolved_repo_root / "docs" / "behavior").resolve(),
        )
        _require_success(
            "portable archive build",
            archive_main(
                [
                    "--runs-dir",
                    _DEFAULT_RUNS_DIR,
                    "--figures-dir",
                    _DEFAULT_FIGURES_DIR,
                    "--hotpot-portable-dir",
                    _DEFAULT_HOTPOT_PORTABLE_OUT,
                    "--conformance-dir",
                    "artifacts/conformance",
                ]
            ),
        )
        _require_success(
            "portable services shutdown",
            services_main(["down", "--profile", "portable"]),
        )
    return {
        "mode": "portable-finalize",
        "repo_root": str(resolved_repo_root),
        "runs_dir": str(runs_dir),
        "figures_dir": str(figures_dir),
    }


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description="High-level workflow commands for the portable lane.")
    parser.add_argument("phase", choices=["setup", "data", "finalize"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    try:
        if args.phase == "setup":
            summary = portable_setup(repo_root=repo_root)
        elif args.phase == "data":
            summary = portable_data(repo_root=repo_root)
        else:
            summary = portable_finalize(repo_root=repo_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
