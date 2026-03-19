"""Prefetch required benchmark datasets into the shared repository cache."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

import numpy as np
import yaml

from maxionbench.orchestration.config_schema import expand_env_placeholders

DEFAULT_ENV_SH = "artifacts/prefetch/dataset_env.sh"
DEFAULT_D3_TARGET = "data/d3/laion_d3_vectors.npy"
DEFAULT_PROCESSED_D3_BASE = "dataset/processed/D3/yfcc-10M/base.npy"
DEFAULT_D4_BEIR_ROOT = "data/beir"
DEFAULT_DOWNLOADED_D4_BEIR_ROOT = "dataset/D4/beir"
DEFAULT_D4_CRAG_TARGET = "data/crag_task_1_and_2_dev_v4.jsonl.bz2"
DEFAULT_DOWNLOADED_D4_CRAG_TARGET = "dataset/D4/crag/crag_task_1_and_2_dev_v4.jsonl.bz2"
DEFAULT_D4_CRAG_URL = "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"

CPU_SCENARIOS = (
    "configs/scenarios/s1_ann_frontier_d3.yaml",
    "configs/scenarios/s2_filtered_ann.yaml",
    "configs/scenarios/s3_churn_smooth.yaml",
    "configs/scenarios/s3b_churn_bursty.yaml",
    "configs/scenarios/s4_hybrid.yaml",
    "configs/scenarios/s6_fusion.yaml",
)
GPU_SCENARIOS = (
    "configs/scenarios/s5_rerank.yaml",
)
CALIBRATE_CONFIG = "configs/scenarios/calibrate_d3.yaml"


@dataclass(frozen=True)
class PrefetchRequirements:
    need_d3: bool
    need_d4_beir: bool
    need_d4_crag: bool
    beir_subsets: tuple[str, ...]
    selected_configs: tuple[str, ...]


def resolve_selected_configs(
    *,
    repo_root: Path,
    scenario_config_dir: str | None = None,
    include_gpu: bool = True,
    skip_s6: bool = False,
) -> list[Path]:
    selected: list[Path] = []
    scenario_root = Path(str(scenario_config_dir)).expanduser() if scenario_config_dir else None
    candidates = [CALIBRATE_CONFIG, *CPU_SCENARIOS]
    if include_gpu:
        candidates.extend(GPU_SCENARIOS)
    if skip_s6:
        candidates = [item for item in candidates if Path(item).name != "s6_fusion.yaml"]
    for rel_path in candidates:
        default_path = (repo_root / rel_path).resolve()
        chosen = default_path
        if scenario_root is not None:
            override_path = scenario_root / Path(rel_path).name
            override_abs = override_path if override_path.is_absolute() else (repo_root / override_path)
            if override_abs.exists():
                chosen = override_abs.resolve()
        if chosen.exists():
            selected.append(chosen)
    return selected


def collect_prefetch_requirements(*, config_paths: Sequence[Path]) -> PrefetchRequirements:
    need_d3 = False
    need_d4_beir = False
    need_d4_crag = False
    beir_subsets: list[str] = []
    selected_configs = [str(path) for path in config_paths]
    for path in config_paths:
        payload = _load_config_mapping(path)
        dataset_bundle = str(payload.get("dataset_bundle", "")).upper()
        if dataset_bundle == "D3":
            if payload.get("dataset_path") or bool(payload.get("calibration_require_real_data", False)):
                need_d3 = True
        if dataset_bundle == "D4" and bool(payload.get("d4_use_real_data", False)):
            if payload.get("d4_beir_root"):
                need_d4_beir = True
                for subset in payload.get("d4_beir_subsets", []) or []:
                    item = str(subset).strip()
                    if item and item not in beir_subsets:
                        beir_subsets.append(item)
            if bool(payload.get("d4_include_crag", False)) and payload.get("d4_crag_path"):
                need_d4_crag = True
    return PrefetchRequirements(
        need_d3=need_d3,
        need_d4_beir=need_d4_beir,
        need_d4_crag=need_d4_crag,
        beir_subsets=tuple(beir_subsets),
        selected_configs=tuple(selected_configs),
    )


def prefetch_for_configs(
    *,
    repo_root: Path,
    config_paths: Sequence[Path],
    env_sh_path: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    requirements = collect_prefetch_requirements(config_paths=config_paths)
    return prefetch_required_datasets(
        repo_root=repo_root,
        requirements=requirements,
        env_sh_path=env_sh_path,
        env=env,
    )


def prefetch_required_datasets(
    *,
    repo_root: Path,
    requirements: PrefetchRequirements,
    env_sh_path: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(env or {})
    dataset_root = _resolve_dataset_root(repo_root=repo_root, env_map=env_map)
    exports: dict[str, str] = {}
    summary: dict[str, Any] = {
        "repo_root": str(repo_root.resolve()),
        "selected_configs": list(requirements.selected_configs),
        "requirements": {
            "need_d3": requirements.need_d3,
            "need_d4_beir": requirements.need_d4_beir,
            "need_d4_crag": requirements.need_d4_crag,
            "beir_subsets": list(requirements.beir_subsets),
        },
        "fetched": {},
    }
    repo_root = repo_root.resolve()

    if requirements.need_d3:
        d3_target = (repo_root / DEFAULT_D3_TARGET).resolve()
        d3_source = _env_value(env_map, "MAXIONBENCH_PREFETCH_D3_SOURCE")
        d3_meta = _ensure_d3_dataset(
            target_path=d3_target,
            source=d3_source,
            existing_candidates=_existing_d3_dataset_candidates(
                repo_root=repo_root,
                dataset_root=dataset_root,
                env_map=env_map,
            ),
        )
        exports["MAXIONBENCH_D3_DATASET_PATH"] = d3_meta["path"]
        exports["MAXIONBENCH_D3_DATASET_SHA256"] = d3_meta["sha256"]
        summary["fetched"]["d3"] = d3_meta

    if requirements.need_d4_beir:
        beir_target_root = (repo_root / DEFAULT_D4_BEIR_ROOT).resolve()
        beir_source = _env_value(env_map, "MAXIONBENCH_PREFETCH_D4_BEIR_SOURCE")
        beir_meta = _ensure_d4_beir_root(
            target_root=beir_target_root,
            required_subsets=requirements.beir_subsets,
            source=beir_source,
            existing_candidates=_existing_d4_beir_candidates(
                repo_root=repo_root,
                dataset_root=dataset_root,
                env_map=env_map,
            ),
        )
        exports["MAXIONBENCH_D4_BEIR_ROOT"] = str(beir_target_root)
        summary["fetched"]["d4_beir"] = beir_meta

    if requirements.need_d4_crag:
        crag_target = (repo_root / DEFAULT_D4_CRAG_TARGET).resolve()
        crag_source = _env_value(env_map, "MAXIONBENCH_PREFETCH_D4_CRAG_SOURCE") or DEFAULT_D4_CRAG_URL
        crag_meta = _ensure_binary_target(
            target_path=crag_target,
            source=crag_source,
            label="D4 CRAG",
            existing_candidates=_existing_d4_crag_candidates(
                repo_root=repo_root,
                dataset_root=dataset_root,
                env_map=env_map,
            ),
        )
        exports["MAXIONBENCH_D4_CRAG_PATH"] = str(crag_target)
        exports["MAXIONBENCH_D4_CRAG_SHA256"] = crag_meta["sha256"]
        summary["fetched"]["d4_crag"] = crag_meta

    _write_env_exports(env_sh_path=env_sh_path, exports=exports)
    summary["env_sh_path"] = str(env_sh_path.resolve())
    summary["exports"] = exports
    return summary


def _load_config_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return dict(expand_env_placeholders(payload))


def _ensure_d3_dataset(
    *,
    target_path: Path,
    source: str | None,
    existing_candidates: Sequence[tuple[Path, str]] = (),
) -> dict[str, Any]:
    if target_path.exists():
        return {
            "path": str(target_path),
            "sha256": _sha256_file(target_path),
            "source": "cache_hit",
        }
    for candidate_path, candidate_source in existing_candidates:
        if not candidate_path.exists():
            continue
        return {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "source": candidate_source,
        }
    if not source:
        raise FileNotFoundError(
            "D3 real data is required but missing. Provide MAXIONBENCH_D3_DATASET_PATH as an existing "
            ".npy/.npz file, or provide MAXIONBENCH_PREFETCH_D3_SOURCE as a local .npy/.npz path or "
            "downloadable URL."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="maxionbench_d3_prefetch_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        acquired = _acquire_source_path(source=source, workdir=tmpdir)
        suffix = acquired.suffix.lower()
        if suffix == ".npy":
            shutil.copy2(acquired, target_path)
        elif suffix == ".npz":
            npz = np.load(acquired, allow_pickle=False)
            if "vectors" not in npz:
                raise ValueError("D3 npz source must contain a `vectors` array")
            np.save(target_path, np.asarray(npz["vectors"], dtype=np.float32))
        else:
            raise ValueError("D3 source must be a .npy or .npz file")
    return {
        "path": str(target_path),
        "sha256": _sha256_file(target_path),
        "source": source,
    }


def _existing_d3_dataset_candidates(
    *,
    repo_root: Path,
    dataset_root: Path,
    env_map: Mapping[str, str],
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    raw_env_path = _env_value(env_map, "MAXIONBENCH_D3_DATASET_PATH")
    if raw_env_path:
        resolved = _resolve_path_from_env(raw_value=raw_env_path, repo_root=repo_root)
        if resolved not in seen:
            candidates.append((resolved, "env_MAXIONBENCH_D3_DATASET_PATH"))
            seen.add(resolved)
    processed_base = (dataset_root / "processed" / "D3" / "yfcc-10M" / "base.npy").resolve()
    if processed_base not in seen:
        candidates.append((processed_base, "processed_dataset_cache"))
        seen.add(processed_base)
    legacy_target = (repo_root / DEFAULT_PROCESSED_D3_BASE).resolve()
    if legacy_target not in seen:
        candidates.append((legacy_target, "processed_dataset_cache"))
    return candidates


def _existing_d4_beir_candidates(
    *,
    repo_root: Path,
    dataset_root: Path,
    env_map: Mapping[str, str],
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    raw_env_path = _env_value(env_map, "MAXIONBENCH_D4_BEIR_ROOT")
    if raw_env_path:
        resolved = _resolve_path_from_env(raw_value=raw_env_path, repo_root=repo_root)
        if resolved not in seen:
            candidates.append((resolved, "env_MAXIONBENCH_D4_BEIR_ROOT"))
            seen.add(resolved)
    downloaded_root = (dataset_root / "D4" / "beir").resolve()
    if downloaded_root not in seen:
        candidates.append((downloaded_root, "downloaded_dataset_cache"))
        seen.add(downloaded_root)
    legacy_root = (repo_root / DEFAULT_DOWNLOADED_D4_BEIR_ROOT).resolve()
    if legacy_root not in seen:
        candidates.append((legacy_root, "downloaded_dataset_cache"))
    return candidates


def _existing_d4_crag_candidates(
    *,
    repo_root: Path,
    dataset_root: Path,
    env_map: Mapping[str, str],
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    raw_env_path = _env_value(env_map, "MAXIONBENCH_D4_CRAG_PATH")
    if raw_env_path:
        resolved = _resolve_path_from_env(raw_value=raw_env_path, repo_root=repo_root)
        if resolved not in seen:
            candidates.append((resolved, "env_MAXIONBENCH_D4_CRAG_PATH"))
            seen.add(resolved)
    downloaded_target = (dataset_root / "D4" / "crag" / Path(DEFAULT_D4_CRAG_TARGET).name).resolve()
    if downloaded_target not in seen:
        candidates.append((downloaded_target, "downloaded_dataset_cache"))
        seen.add(downloaded_target)
    legacy_target = (repo_root / DEFAULT_DOWNLOADED_D4_CRAG_TARGET).resolve()
    if legacy_target not in seen:
        candidates.append((legacy_target, "downloaded_dataset_cache"))
    return candidates


def _resolve_dataset_root(*, repo_root: Path, env_map: Mapping[str, str]) -> Path:
    raw = _env_value(env_map, "MAXIONBENCH_DATASET_ROOT") or "dataset"
    return _resolve_path_from_env(raw_value=raw, repo_root=repo_root)


def _resolve_path_from_env(*, raw_value: str, repo_root: Path) -> Path:
    candidate = Path(str(raw_value)).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    return candidate.resolve()


def _ensure_d4_beir_root(
    *,
    target_root: Path,
    required_subsets: Sequence[str],
    source: str | None,
    existing_candidates: Sequence[tuple[Path, str]] = (),
) -> dict[str, Any]:
    missing = _missing_beir_subsets(root=target_root, required_subsets=required_subsets)
    if not missing:
        return {
            "target_root": str(target_root),
            "required_subsets": list(required_subsets),
            "source": "cache_hit",
        }
    for candidate_root, candidate_source in existing_candidates:
        if candidate_root.resolve() == target_root.resolve():
            continue
        if _missing_beir_subsets(root=candidate_root, required_subsets=required_subsets):
            continue
        _copy_beir_subsets(
            source_root=candidate_root,
            target_root=target_root,
            required_subsets=required_subsets,
        )
        return {
            "target_root": str(target_root),
            "required_subsets": list(required_subsets),
            "source": candidate_source,
        }
    if not source:
        raise FileNotFoundError(
            "D4 BEIR bundles are required but missing. Provide MAXIONBENCH_PREFETCH_D4_BEIR_SOURCE "
            "as a local directory/archive or downloadable archive URL."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="maxionbench_d4_beir_prefetch_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        acquired = _acquire_source_path(source=source, workdir=tmpdir)
        extracted_root = _materialize_directory_source(acquired=acquired, workdir=tmpdir)
        _copy_beir_subsets(
            source_root=extracted_root,
            target_root=target_root,
            required_subsets=required_subsets,
            source_label=source,
        )
    missing_after = _missing_beir_subsets(root=target_root, required_subsets=required_subsets)
    if missing_after:
        raise FileNotFoundError(f"Missing BEIR subset(s) after prefetch: {missing_after}")
    return {
        "target_root": str(target_root),
        "required_subsets": list(required_subsets),
        "source": source,
    }


def _copy_beir_subsets(
    *,
    source_root: Path,
    target_root: Path,
    required_subsets: Sequence[str],
    source_label: str | None = None,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for subset in required_subsets:
        subset_source = _find_named_directory(root=source_root, name=subset)
        if subset_source is None:
            label = source_label or str(source_root)
            raise FileNotFoundError(f"Unable to locate BEIR subset `{subset}` in {label}")
        destination = target_root / subset
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(subset_source, destination)


def _ensure_binary_target(
    *,
    target_path: Path,
    source: str | None,
    label: str,
    existing_candidates: Sequence[tuple[Path, str]] = (),
) -> dict[str, Any]:
    if target_path.exists():
        return {
            "target_path": str(target_path),
            "sha256": _sha256_file(target_path),
            "source": "cache_hit",
        }
    for candidate_path, candidate_source in existing_candidates:
        if candidate_path.resolve() == target_path.resolve():
            continue
        if not candidate_path.exists() or candidate_path.is_dir():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, target_path)
        return {
            "target_path": str(target_path),
            "sha256": _sha256_file(target_path),
            "source": candidate_source,
        }
    if not source:
        raise FileNotFoundError(f"{label} is required but missing; provide an explicit source path or URL.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="maxionbench_binary_prefetch_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        acquired = _acquire_source_path(source=source, workdir=tmpdir)
        if acquired.is_dir():
            raise ValueError(f"{label} source must resolve to a file, got directory: {source}")
        shutil.copy2(acquired, target_path)
    return {
        "target_path": str(target_path),
        "sha256": _sha256_file(target_path),
        "source": source,
    }


def _missing_beir_subsets(*, root: Path, required_subsets: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for subset in required_subsets:
        subset_dir = root / str(subset)
        if not (subset_dir / "corpus.jsonl").exists():
            missing.append(str(subset))
            continue
        if not (subset_dir / "queries.jsonl").exists():
            missing.append(str(subset))
            continue
        if not (subset_dir / "qrels" / "test.tsv").exists():
            missing.append(str(subset))
    return missing


def _materialize_directory_source(*, acquired: Path, workdir: Path) -> Path:
    if acquired.is_dir():
        return acquired
    name = acquired.name.lower()
    extract_root = (workdir / "extracted").resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(acquired):
        with zipfile.ZipFile(acquired) as archive:
            archive.extractall(extract_root)
        return extract_root
    if tarfile.is_tarfile(acquired):
        with tarfile.open(acquired) as archive:
            archive.extractall(extract_root)
        return extract_root
    if name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
        shutil.unpack_archive(str(acquired), str(extract_root))
        return extract_root
    raise ValueError(f"Directory source must be a directory or archive, got: {acquired}")


def _find_named_directory(*, root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = sorted(path for path in root.rglob(name) if path.is_dir())
    if not matches:
        return None
    return matches[0]


def _acquire_source_path(*, source: str, workdir: Path) -> Path:
    if _is_url(source):
        filename = _url_filename(source) or "download.bin"
        destination = (workdir / filename).resolve()
        _download_url(url=source, destination=destination)
        return destination
    candidate = Path(source).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"source path does not exist: {source}")
    return resolved


def _download_url(*, url: str, destination: Path) -> None:
    with urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"}


def _url_filename(url: str) -> str:
    parsed = urlparse(url)
    return Path(parsed.path).name


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    value = str(env.get(key, "")).strip()
    return value or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_env_exports(*, env_sh_path: Path, exports: Mapping[str, str]) -> None:
    env_sh_path = env_sh_path.resolve()
    env_sh_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key in sorted(exports):
        lines.append(f"export {key}={shlex.quote(str(exports[key]))}")
    env_sh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Prefetch required MaxionBench datasets into the shared cache")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--scenario-config-dir", default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-s6", action="store_true")
    parser.add_argument("--env-sh", default=DEFAULT_ENV_SH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(str(args.repo_root)).expanduser().resolve()
    env_sh = Path(str(args.env_sh))
    if not env_sh.is_absolute():
        env_sh = (repo_root / env_sh).resolve()

    config_paths = resolve_selected_configs(
        repo_root=repo_root,
        scenario_config_dir=str(args.scenario_config_dir) if args.scenario_config_dir else None,
        include_gpu=not bool(args.skip_gpu),
        skip_s6=bool(args.skip_s6),
    )
    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=config_paths,
        env_sh_path=env_sh,
        env=os.environ,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"prefetched datasets into {summary['env_sh_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
