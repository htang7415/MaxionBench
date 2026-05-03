"""Build manifest-driven local run matrices."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


_DATASET_ROOT_TOKEN = "${MAXIONBENCH_DATASET_ROOT:-dataset}"
_DEFAULT_OUTPUT_ROOT = "artifacts/runs/portable_matrix"
_SCENARIO_ORDER = {
    "s1_single_hop": 0,
    "s2_streaming_memory": 1,
    "s3_multi_hop": 2,
}
_PORTABLE_EMBEDDING_VARIANTS = (
    ("bge-small-en-v1-5", "BAAI/bge-small-en-v1.5", 384),
    ("bge-base-en-v1-5", "BAAI/bge-base-en-v1.5", 768),
)


@dataclass(frozen=True)
class RunMatrixRow:
    group: str
    config_path: str
    engine: str
    scenario: str
    dataset_bundle: str
    template_name: str


@dataclass(frozen=True)
class RunMatrix:
    repo_root: str
    generated_config_dir: str
    output_root: str
    budget_level: str | None
    cpu_rows: list[RunMatrixRow]
    gpu_rows: list[RunMatrixRow]
    selected_engines: list[str]
    selected_templates: list[str]
    lane: str

    def iter_rows(self, *, lane: str | None = None) -> Iterable[RunMatrixRow]:
        selected_lane = (lane or self.lane).strip().lower()
        if selected_lane == "cpu":
            return tuple(self.cpu_rows)
        if selected_lane == "gpu":
            return tuple(self.gpu_rows)
        return tuple(self.cpu_rows) + tuple(self.gpu_rows)


def build_run_matrix(
    *,
    repo_root: Path,
    scenario_config_dir: Path,
    engine_config_dir: Path,
    out_dir: Path,
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    budget_level: str | None = None,
    lane: str = "all",
    skip_s6: bool = False,
) -> RunMatrix:
    normalized_lane = _normalize_lane(lane)
    normalized_budget = _normalize_budget_level(budget_level)
    resolved_repo_root = repo_root.expanduser().resolve()
    resolved_scenarios = _resolve_dir(path=scenario_config_dir, repo_root=resolved_repo_root)
    resolved_engines = _resolve_dir(path=engine_config_dir, repo_root=resolved_repo_root)
    resolved_out_dir = out_dir.expanduser().resolve()
    generated_config_dir = resolved_out_dir / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    templates = _load_templates(resolved_scenarios, skip_s6=skip_s6)
    engines = _load_engine_payloads(resolved_engines)

    cpu_rows: list[RunMatrixRow] = []
    gpu_rows: list[RunMatrixRow] = []
    selected_templates: list[str] = []
    selected_engines: list[str] = []

    for template_name, scenario_payload in templates:
        expanded_templates = _expand_template_variants(template_name=template_name, payload=scenario_payload)
        selected_templates.extend(name for name, _ in expanded_templates)
        for expanded_template_name, expanded_payload in expanded_templates:
            for engine_name, engine_payload in engines:
                merged = _compose_config(
                    scenario_payload=expanded_payload,
                    engine_payload=engine_payload,
                    template_name=expanded_template_name,
                    output_root=output_root,
                    budget_level=normalized_budget,
                )
                group = _task_group_for_payload(merged=merged, template_name=expanded_template_name)
                if normalized_lane == "cpu" and group == "gpu":
                    continue
                if normalized_lane == "gpu" and group != "gpu":
                    continue
                if engine_name not in selected_engines:
                    selected_engines.append(engine_name)
                target_path = generated_config_dir / group / f"{Path(expanded_template_name).stem}__{_slug(engine_name)}.yaml"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
                row = RunMatrixRow(
                    group=group,
                    config_path=str(target_path.resolve()),
                    engine=str(merged.get("engine", engine_name)),
                    scenario=str(merged.get("scenario", "")),
                    dataset_bundle=str(merged.get("dataset_bundle", "")),
                    template_name=expanded_template_name,
                )
                if group == "gpu":
                    gpu_rows.append(row)
                else:
                    cpu_rows.append(row)

    cpu_rows.sort(key=_row_sort_key)
    gpu_rows.sort(key=_row_sort_key)
    matrix = RunMatrix(
        repo_root=str(resolved_repo_root),
        generated_config_dir=str(generated_config_dir.resolve()),
        output_root=str(output_root),
        budget_level=normalized_budget,
        cpu_rows=cpu_rows,
        gpu_rows=gpu_rows,
        selected_engines=selected_engines,
        selected_templates=selected_templates,
        lane=normalized_lane,
    )
    matrix_path = resolved_out_dir / "run_matrix.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(
        json.dumps(
            {
                "repo_root": matrix.repo_root,
                "generated_config_dir": matrix.generated_config_dir,
                "output_root": matrix.output_root,
                "budget_level": matrix.budget_level,
                "cpu_rows": [asdict(row) for row in matrix.cpu_rows],
                "gpu_rows": [asdict(row) for row in matrix.gpu_rows],
                "selected_engines": list(matrix.selected_engines),
                "selected_templates": list(matrix.selected_templates),
                "lane": matrix.lane,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix


def load_run_matrix(path: Path) -> RunMatrix:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return RunMatrix(
        repo_root=str(payload["repo_root"]),
        generated_config_dir=str(payload["generated_config_dir"]),
        output_root=str(payload.get("output_root", _DEFAULT_OUTPUT_ROOT)),
        budget_level=_normalize_budget_level(payload.get("budget_level")),
        cpu_rows=[RunMatrixRow(**row) for row in payload.get("cpu_rows", [])],
        gpu_rows=[RunMatrixRow(**row) for row in payload.get("gpu_rows", [])],
        selected_engines=[str(item) for item in payload.get("selected_engines", [])],
        selected_templates=[str(item) for item in payload.get("selected_templates", [])],
        lane=_normalize_lane(str(payload.get("lane", "all"))),
    )


def _normalize_lane(lane: str) -> str:
    normalized = str(lane).strip().lower()
    if normalized not in {"cpu", "gpu", "all"}:
        raise ValueError(f"lane must be one of cpu/gpu/all, got {lane!r}")
    return normalized


def _normalize_budget_level(budget_level: object | None) -> str | None:
    if budget_level is None:
        return None
    normalized = str(budget_level).strip().lower()
    if not normalized:
        return None
    if normalized not in {"b0", "b1", "b2"}:
        raise ValueError(f"budget_level must be one of b0/b1/b2, got {budget_level!r}")
    return normalized


def _resolve_dir(*, path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repo_relative = (repo_root / candidate).resolve()
    if repo_relative.exists():
        return repo_relative
    return candidate.resolve()


def _load_templates(root: Path, *, skip_s6: bool) -> list[tuple[str, dict[str, Any]]]:
    del skip_s6
    templates: list[tuple[str, dict[str, Any]]] = []
    for path in _yaml_config_paths(root):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"scenario template must be a mapping: {path}")
        templates.append((path.name, dict(payload)))
    templates.sort(key=lambda item: _template_sort_key(template_name=item[0], payload=item[1]))
    return templates


def _expand_template_variants(*, template_name: str, payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if str(payload.get("profile", "")).strip().lower() not in {"maxionbench", "portable-agentic"}:
        return [(template_name, dict(payload))]
    stem = Path(template_name).stem
    variants: list[tuple[str, dict[str, Any]]] = []
    for suffix, model_id, dim in _PORTABLE_EMBEDDING_VARIANTS:
        variant = dict(payload)
        variant["embedding_model"] = model_id
        variant["embedding_dim"] = dim
        variant["vector_dim"] = dim
        variants.append((f"{stem}__{suffix}.yaml", variant))
    return variants


def _load_engine_payloads(root: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in _yaml_config_paths(root):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"engine config must be a mapping: {path}")
        engine_name = str(payload.get("engine", "")).strip()
        if not engine_name:
            raise ValueError(f"engine config must define `engine`: {path}")
        payloads.append((engine_name, dict(payload)))
    return payloads


def _yaml_config_paths(root: Path) -> list[Path]:
    return [path for path in sorted(root.glob("*.yaml")) if not path.name.startswith("._")]


def _compose_config(
    *,
    scenario_payload: dict[str, Any],
    engine_payload: dict[str, Any],
    template_name: str,
    output_root: str,
    budget_level: str | None,
) -> dict[str, Any]:
    merged = dict(scenario_payload)
    if budget_level is not None:
        merged["budget_level"] = budget_level
    merged["engine"] = str(engine_payload.get("engine", merged.get("engine", "mock")))
    merged["engine_version"] = str(engine_payload.get("engine_version", merged.get("engine_version", "0.1.0")))
    adapter_options = dict(merged.get("adapter_options") or {})
    adapter_options.update(dict(engine_payload.get("adapter_options") or {}))
    merged["adapter_options"] = adapter_options
    merged["output_dir"] = f"{str(output_root).rstrip('/')}/{Path(template_name).stem}/{_slug(str(merged['engine']))}"
    _normalize_pipeline_dataset_refs(payload=merged, template_name=template_name)
    return merged


def _normalize_pipeline_dataset_refs(*, payload: dict[str, Any], template_name: str) -> None:
    del template_name
    bundle = str(payload.get("dataset_bundle", "")).upper()
    if bundle == "D4":
        payload["processed_dataset_path"] = f"{_DATASET_ROOT_TOKEN}/processed/D4"
    if bundle == "HOTPOT_PORTABLE":
        payload["processed_dataset_path"] = f"{_DATASET_ROOT_TOKEN}/processed/hotpot_portable"


def _task_group_for_payload(*, merged: dict[str, Any], template_name: str) -> str:
    del merged
    template = Path(template_name).stem.lower()
    if "track_b" in template or "track_c" in template:
        return "gpu"
    return "cpu"


def _row_sort_key(row: RunMatrixRow) -> tuple[int, str, str]:
    template_key = _template_sort_key(
        template_name=row.template_name,
        scenario=row.scenario,
        dataset_bundle=row.dataset_bundle,
    )
    return (*template_key, row.engine)


def _template_sort_key(
    *,
    template_name: str,
    payload: dict[str, Any] | None = None,
    scenario: str | None = None,
    dataset_bundle: str | None = None,
) -> tuple[int, int, str]:
    stem = Path(template_name).stem
    resolved_scenario = str((payload or {}).get("scenario", scenario or "")).strip().lower()
    del dataset_bundle
    scenario_order = _SCENARIO_ORDER.get(stem, _SCENARIO_ORDER.get(resolved_scenario, 999))
    return (scenario_order, 0, stem)


def _slug(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Build a MaxionBench run matrix from scenario and engine configs")
    parser.add_argument("--scenario-config-dir", default="configs/scenarios_portable")
    parser.add_argument("--engine-config-dir", default="configs/engines_portable")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--budget", default=None, choices=["b0", "b1", "b2"])
    parser.add_argument("--lane", default="all", choices=["cpu", "gpu", "all"])
    parser.add_argument("--skip-s6", action="store_true", help="Deprecated no-op retained for old scripts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    matrix = build_run_matrix(
        repo_root=repo_root,
        scenario_config_dir=Path(args.scenario_config_dir),
        engine_config_dir=Path(args.engine_config_dir),
        out_dir=Path(args.out_dir),
        output_root=str(args.output_root),
        budget_level=args.budget,
        lane=str(args.lane),
        skip_s6=bool(args.skip_s6),
    )
    summary = {
        "repo_root": matrix.repo_root,
        "generated_config_dir": matrix.generated_config_dir,
        "output_root": matrix.output_root,
        "budget_level": matrix.budget_level,
        "lane": matrix.lane,
        "cpu_rows": len(matrix.cpu_rows),
        "gpu_rows": len(matrix.gpu_rows),
        "selected_engines": list(matrix.selected_engines),
        "selected_templates": list(matrix.selected_templates),
        "matrix_path": str((Path(args.out_dir).expanduser().resolve() / "run_matrix.json").resolve()),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
