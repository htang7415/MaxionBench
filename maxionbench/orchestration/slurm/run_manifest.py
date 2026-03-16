"""Build and resolve manifest-driven Slurm run matrices."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml


_DATASET_ROOT_TOKEN = "${MAXIONBENCH_DATASET_ROOT:-dataset}"
_OUTPUT_ROOT_TOKEN = "${MAXIONBENCH_OUTPUT_ROOT:-artifacts/runs/paper_matrix}"
_SCENARIO_ORDER = {
    "s1_ann_frontier": 0,
    "s1_ann_frontier_d3": 1,
    "s2_filtered_ann": 2,
    "s3_churn_smooth": 3,
    "s3b_churn_bursty": 4,
    "s4_hybrid": 5,
    "s5_rerank": 6,
    "s6_fusion": 7,
}


@dataclass(frozen=True)
class RunManifestRow:
    group: str
    config_path: str
    engine: str
    scenario: str
    dataset_bundle: str
    template_name: str


@dataclass(frozen=True)
class RunManifest:
    repo_root: str
    generated_config_dir: str
    cpu_rows: list[RunManifestRow]
    gpu_rows: list[RunManifestRow]
    selected_engines: list[str]
    selected_templates: list[str]


def build_run_manifest(
    *,
    repo_root: Path,
    scenario_config_dir: Path,
    engine_config_dir: Path,
    out_dir: Path,
    include_gpu: bool = True,
    skip_s6: bool = False,
) -> RunManifest:
    resolved_repo_root = repo_root.expanduser().resolve()
    resolved_scenarios = _resolve_dir(path=scenario_config_dir, repo_root=resolved_repo_root)
    resolved_engines = _resolve_dir(path=engine_config_dir, repo_root=resolved_repo_root)
    resolved_out_dir = out_dir.expanduser().resolve()
    generated_config_dir = resolved_out_dir / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    templates = _load_templates(resolved_scenarios, skip_s6=skip_s6)
    engines = _load_engine_payloads(resolved_engines)

    cpu_rows: list[RunManifestRow] = []
    gpu_rows: list[RunManifestRow] = []
    selected_templates: list[str] = []
    selected_engines: list[str] = []

    for template_name, scenario_payload in templates:
        selected_templates.append(template_name)
        for engine_name, engine_payload in engines:
            merged = _compose_config(
                scenario_payload=scenario_payload,
                engine_payload=engine_payload,
                template_name=template_name,
            )
            group = _task_group_for_payload(merged)
            if group == "gpu" and not include_gpu:
                continue
            if engine_name not in selected_engines:
                selected_engines.append(engine_name)
            target_path = generated_config_dir / group / f"{Path(template_name).stem}__{_slug(engine_name)}.yaml"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
            row = RunManifestRow(
                group=group,
                config_path=str(target_path.resolve()),
                engine=str(merged.get("engine", engine_name)),
                scenario=str(merged.get("scenario", "")),
                dataset_bundle=str(merged.get("dataset_bundle", "")),
                template_name=template_name,
            )
            if group == "gpu":
                gpu_rows.append(row)
            else:
                cpu_rows.append(row)

    cpu_rows.sort(key=_row_sort_key)
    gpu_rows.sort(key=_row_sort_key)
    manifest = RunManifest(
        repo_root=str(resolved_repo_root),
        generated_config_dir=str(generated_config_dir.resolve()),
        cpu_rows=cpu_rows,
        gpu_rows=gpu_rows,
        selected_engines=selected_engines,
        selected_templates=selected_templates,
    )
    manifest_path = resolved_out_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "repo_root": manifest.repo_root,
                "generated_config_dir": manifest.generated_config_dir,
                "cpu_rows": [asdict(row) for row in manifest.cpu_rows],
                "gpu_rows": [asdict(row) for row in manifest.gpu_rows],
                "selected_engines": list(manifest.selected_engines),
                "selected_templates": list(manifest.selected_templates),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_run_manifest(path: Path) -> RunManifest:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return RunManifest(
        repo_root=str(payload["repo_root"]),
        generated_config_dir=str(payload["generated_config_dir"]),
        cpu_rows=[RunManifestRow(**row) for row in payload.get("cpu_rows", [])],
        gpu_rows=[RunManifestRow(**row) for row in payload.get("gpu_rows", [])],
        selected_engines=[str(item) for item in payload.get("selected_engines", [])],
        selected_templates=[str(item) for item in payload.get("selected_templates", [])],
    )


def resolve_manifest_row(*, manifest_path: Path, group: str, task_id: int) -> RunManifestRow:
    manifest = load_run_manifest(manifest_path)
    rows = manifest.cpu_rows if str(group).strip().lower() == "cpu" else manifest.gpu_rows
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"task id {task_id} is out of range for {group} rows ({len(rows)})")
    return rows[task_id]


def _resolve_dir(*, path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repo_relative = (repo_root / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    return candidate.resolve()


def _load_templates(root: Path, *, skip_s6: bool) -> list[tuple[str, dict[str, Any]]]:
    templates: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name == "calibrate_d3.yaml":
            continue
        if skip_s6 and path.stem == "s6_fusion":
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"scenario template must be a mapping: {path}")
        templates.append((path.name, dict(payload)))
    return templates


def _load_engine_payloads(root: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"engine config must be a mapping: {path}")
        engine_name = str(payload.get("engine", "")).strip()
        if not engine_name:
            raise ValueError(f"engine config must define `engine`: {path}")
        payloads.append((engine_name, dict(payload)))
    return payloads


def _compose_config(
    *,
    scenario_payload: dict[str, Any],
    engine_payload: dict[str, Any],
    template_name: str,
) -> dict[str, Any]:
    merged = dict(scenario_payload)
    merged["engine"] = str(engine_payload.get("engine", merged.get("engine", "mock")))
    merged["engine_version"] = str(engine_payload.get("engine_version", merged.get("engine_version", "0.1.0")))
    adapter_options = dict(merged.get("adapter_options") or {})
    adapter_options.update(dict(engine_payload.get("adapter_options") or {}))
    merged["adapter_options"] = adapter_options
    merged["output_dir"] = f"{_OUTPUT_ROOT_TOKEN}/{Path(template_name).stem}/{_slug(str(merged['engine']))}"
    _normalize_pipeline_dataset_refs(payload=merged, template_name=template_name)
    return merged


def _normalize_pipeline_dataset_refs(*, payload: dict[str, Any], template_name: str) -> None:
    bundle = str(payload.get("dataset_bundle", "")).upper()
    scenario = str(payload.get("scenario", "")).strip().lower()
    if bundle == "D4":
        payload["processed_dataset_path"] = f"{_DATASET_ROOT_TOKEN}/processed/D4"
        return
    if bundle != "D3":
        return

    processed_d3_root = f"{_DATASET_ROOT_TOKEN}/processed/D3/yfcc-10M"
    if scenario == "calibrate_d3":
        payload["processed_dataset_path"] = processed_d3_root
        payload.pop("dataset_path", None)
        payload.pop("dataset_path_sha256", None)
        return
    if scenario in {"s3_churn_smooth", "s3b_churn_bursty"}:
        payload["processed_dataset_path"] = processed_d3_root
        payload.pop("dataset_path", None)
        payload.pop("dataset_path_sha256", None)
        return
    if Path(template_name).stem == "s1_ann_frontier_d3":
        payload["processed_dataset_path"] = processed_d3_root
        payload.pop("dataset_path", None)
        payload.pop("dataset_path_sha256", None)
        return
    payload["dataset_path"] = f"{processed_d3_root}/base.npy"
    payload.pop("dataset_path_sha256", None)


def _task_group_for_payload(payload: dict[str, Any]) -> str:
    scenario = str(payload.get("scenario", "")).strip().lower()
    engine = _slug(str(payload.get("engine", "")))
    if scenario == "s5_rerank":
        return "gpu"
    if engine == "faiss_gpu":
        return "gpu"
    return "cpu"


def _row_sort_key(row: RunManifestRow) -> tuple[int, str, str]:
    return (
        _SCENARIO_ORDER.get(row.template_name.removesuffix(".yaml"), 999),
        row.template_name,
        row.engine,
    )


def _slug(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Build or resolve MaxionBench Slurm run manifests")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build a run manifest from engine configs and scenario templates")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--scenario-config-dir", required=True)
    build.add_argument("--engine-config-dir", default="configs/engines")
    build.add_argument("--out-dir", required=True)
    build.add_argument("--skip-gpu", action="store_true")
    build.add_argument("--skip-s6", action="store_true")
    build.add_argument("--json", action="store_true")

    resolve = sub.add_parser("resolve", help="Resolve one manifest row for a Slurm array task")
    resolve.add_argument("--manifest", required=True)
    resolve.add_argument("--group", required=True, choices=["cpu", "gpu"])
    resolve.add_argument("--task-id", type=int, required=True)
    resolve.add_argument("--field", default="config_path")
    resolve.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        manifest = build_run_manifest(
            repo_root=Path(args.repo_root),
            scenario_config_dir=Path(args.scenario_config_dir),
            engine_config_dir=Path(args.engine_config_dir),
            out_dir=Path(args.out_dir),
            include_gpu=not bool(args.skip_gpu),
            skip_s6=bool(args.skip_s6),
        )
        payload = {
            "repo_root": manifest.repo_root,
            "generated_config_dir": manifest.generated_config_dir,
            "cpu_count": len(manifest.cpu_rows),
            "gpu_count": len(manifest.gpu_rows),
            "selected_engines": manifest.selected_engines,
            "selected_templates": manifest.selected_templates,
            "manifest_path": str((Path(args.out_dir).expanduser().resolve() / "run_manifest.json").resolve()),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload["manifest_path"])
        return 0

    row = resolve_manifest_row(
        manifest_path=Path(args.manifest),
        group=str(args.group),
        task_id=int(args.task_id),
    )
    payload = asdict(row)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    value = payload.get(str(args.field))
    if value is None:
        raise ValueError(f"unknown manifest row field: {args.field}")
    print(value)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
