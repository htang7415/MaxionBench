"""Submit a portable-agentic budget run on a local Mac host."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from maxionbench.orchestration.config_schema import load_run_config
from maxionbench.orchestration.run_matrix import RunMatrix, build_run_matrix
from maxionbench.tools.service_lifecycle import services_up
from maxionbench.tools.execute_run_matrix import execute_run_matrix
from maxionbench.tools.verify_promotion_gate import verify_portable_promotion_gate

_PORTABLE_SERVICE_ENGINES = frozenset({"qdrant", "pgvector"})
_DEFAULT_PAPER_EXCLUDED_ENGINES = frozenset({"lancedb-service"})


def _default_lancedb_local_uri(*, repo_root: Path, lane: str) -> str:
    scratch_root = Path(tempfile.gettempdir()).resolve()
    return str((scratch_root / "maxionbench" / repo_root.name / "lancedb" / lane).resolve())


def _ensure_portable_services(
    *,
    repo_root: Path,
    selected_engines: list[str],
    adapter_timeout_s: float,
    poll_interval_s: float,
) -> None:
    required = sorted({engine for engine in selected_engines if engine in _PORTABLE_SERVICE_ENGINES})
    if not required:
        return
    services_up(
        compose_file=(repo_root / "docker-compose.yml").resolve(),
        services=required,
        wait=True,
        timeout_s=adapter_timeout_s,
        poll_interval_s=poll_interval_s,
    )


def _execution_engines(*, selected_engines: list[str], engine_filter: set[str] | None) -> list[str]:
    if not engine_filter:
        return list(selected_engines)
    allowed = {engine.strip() for engine in engine_filter if engine.strip()}
    return [engine for engine in selected_engines if engine in allowed]


def _effective_engine_filter(*, selected_engines: list[str], engine_filter: set[str] | None) -> set[str] | None:
    if engine_filter is not None:
        return {engine.strip() for engine in engine_filter if engine.strip()} or None
    return {
        engine for engine in selected_engines
        if engine not in _DEFAULT_PAPER_EXCLUDED_ENGINES
    }


def _selected_matrix_rows(
    *,
    matrix: RunMatrix,
    lane: str,
    engine_filter: set[str] | None,
    scenario_filter: set[str] | None,
    template_filter: set[str] | None,
    max_runs: int | None,
) -> list[Any]:
    rows = list(matrix.iter_rows(lane=lane))
    filtered: list[Any] = []
    for row in rows:
        if engine_filter and str(row.engine) not in engine_filter:
            continue
        if scenario_filter and str(row.scenario) not in scenario_filter:
            continue
        template_name = str(row.template_name)
        if template_filter and Path(template_name).stem not in template_filter and template_name not in template_filter:
            continue
        filtered.append(row)
        if max_runs is not None and len(filtered) >= max_runs:
            break
    return filtered


def _resolve_repo_relative_path(*, value: str, repo_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _preflight_processed_datasets(*, repo_root: Path, rows: list[Any]) -> None:
    for row in rows:
        config_path = Path(row.config_path).expanduser().resolve()
        cfg = load_run_config(config_path)
        if not cfg.processed_dataset_path:
            continue
        resolved = _resolve_repo_relative_path(value=str(cfg.processed_dataset_path), repo_root=repo_root)
        if resolved.exists():
            continue
        bundle = str(cfg.dataset_bundle).strip().upper()
        if bundle == "HOTPOT_PORTABLE":
            hotpot_root = (repo_root / "dataset" / "D4" / "hotpotqa" / "hotpot_dev_distractor_v1.json").resolve()
            raise FileNotFoundError(
                "portable HotpotQA dataset missing: "
                f"{resolved}. `s3_multi_hop` requires the official HotpotQA dev distractor source at {hotpot_root}, "
                "then rerun `maxionbench portable-workflow data --json`."
            )
        raise FileNotFoundError(f"processed dataset path not found: {resolved}")


def submit_portable(
    *,
    budget: str,
    repo_root: Path,
    scenario_config_dir: Path,
    engine_config_dir: Path,
    out_dir: Path | None = None,
    output_root: Path | None = None,
    lane: str = "cpu",
    skip_s6: bool = True,
    seed: int | None = None,
    repeats: int | None = None,
    no_retry: bool = False,
    skip_completed: bool = True,
    continue_on_failure: bool = False,
    engine_filter: set[str] | None = None,
    scenario_filter: set[str] | None = None,
    template_filter: set[str] | None = None,
    max_runs: int | None = None,
    deadline_hours: float = 24.0,
    adapter_timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
    verify_promotion: bool = True,
) -> dict[str, Any]:
    normalized_budget = str(budget).strip().lower()
    if normalized_budget not in {"b0", "b1", "b2"}:
        raise ValueError(f"budget must be one of b0/b1/b2, got {budget!r}")

    resolved_repo_root = repo_root.expanduser().resolve()
    if not str(os.environ.get("MAXIONBENCH_LANCEDB_INPROC_URI") or "").strip():
        os.environ["MAXIONBENCH_LANCEDB_INPROC_URI"] = _default_lancedb_local_uri(
            repo_root=resolved_repo_root,
            lane="inproc",
        )
    resolved_out_dir = (
        out_dir.expanduser().resolve()
        if out_dir is not None
        else (resolved_repo_root / "artifacts" / "run_matrix" / f"portable_{normalized_budget}").resolve()
    )
    resolved_output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else (resolved_repo_root / "artifacts" / "runs" / "portable" / normalized_budget).resolve()
    )

    matrix = build_run_matrix(
        repo_root=resolved_repo_root,
        scenario_config_dir=scenario_config_dir,
        engine_config_dir=engine_config_dir,
        out_dir=resolved_out_dir,
        output_root=str(resolved_output_root),
        budget_level=normalized_budget,
        lane=lane,
        skip_s6=skip_s6,
    )
    effective_engine_filter = _effective_engine_filter(
        selected_engines=list(matrix.selected_engines),
        engine_filter=engine_filter,
    )
    selected_rows = _selected_matrix_rows(
        matrix=matrix,
        lane=lane,
        engine_filter=effective_engine_filter,
        scenario_filter=scenario_filter,
        template_filter=template_filter,
        max_runs=max_runs,
    )
    _preflight_processed_datasets(
        repo_root=resolved_repo_root,
        rows=selected_rows,
    )
    _ensure_portable_services(
        repo_root=resolved_repo_root,
        selected_engines=_execution_engines(
            selected_engines=list(matrix.selected_engines),
            engine_filter=effective_engine_filter,
        ),
        adapter_timeout_s=adapter_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    matrix_path = resolved_out_dir / "run_matrix.json"
    execution = execute_run_matrix(
        matrix_path=matrix_path,
        lane=lane,
        budget=normalized_budget,
        seed=seed,
        repeats=repeats,
        no_retry=no_retry,
        skip_completed=skip_completed,
        continue_on_failure=continue_on_failure,
        engine_filter=effective_engine_filter,
        scenario_filter=scenario_filter,
        template_filter=template_filter,
        max_runs=max_runs,
        deadline_hours=deadline_hours,
        adapter_timeout_s=adapter_timeout_s,
        poll_interval_s=poll_interval_s,
    )

    summary: dict[str, Any] = {
        "mode": "portable-submit",
        "budget": normalized_budget,
        "matrix_path": str(matrix_path.resolve()),
        "generated_config_dir": str(Path(matrix.generated_config_dir).resolve()),
        "output_root": str(resolved_output_root),
        "selected_engines": _execution_engines(
            selected_engines=list(matrix.selected_engines),
            engine_filter=effective_engine_filter,
        ),
        "excluded_engines": sorted(
            set(matrix.selected_engines)
            - set(
                _execution_engines(
                    selected_engines=list(matrix.selected_engines),
                    engine_filter=effective_engine_filter,
                )
            )
        ),
        "selected_templates": list(matrix.selected_templates),
        "matrix_rows": _matrix_row_counts(matrix=matrix),
        "execution": execution,
    }
    if verify_promotion and normalized_budget in {"b0", "b1"}:
        candidates_path = resolved_out_dir / "promotion_candidates.json"
        summary["promotion"] = verify_portable_promotion_gate(
            results_path=resolved_output_root,
            from_budget=normalized_budget,
            out_candidates_path=candidates_path,
        )
    return summary


def _matrix_row_counts(*, matrix: RunMatrix) -> dict[str, int]:
    return {
        "cpu": len(matrix.cpu_rows),
        "gpu": len(matrix.gpu_rows),
        "all": len(matrix.cpu_rows) + len(matrix.gpu_rows),
    }


def _parse_csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return items or None


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description="Submit a portable-agentic budget run for a local Mac host.")
    parser.add_argument("--budget", required=True, choices=["b0", "b1", "b2"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--scenario-config-dir", default="configs/scenarios_portable")
    parser.add_argument("--engine-config-dir", default="configs/engines_portable")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--lane", default="cpu", choices=["cpu", "gpu", "all"])
    parser.add_argument("--skip-s6", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--engine-filter", default=None)
    parser.add_argument("--scenario-filter", default=None)
    parser.add_argument("--template-filter", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--deadline-hours", type=float, default=24.0)
    parser.add_argument("--adapter-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--no-verify-promotion", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = submit_portable(
        budget=str(args.budget),
        repo_root=Path(args.repo_root),
        scenario_config_dir=Path(args.scenario_config_dir),
        engine_config_dir=Path(args.engine_config_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        output_root=Path(args.output_root) if args.output_root else None,
        lane=str(args.lane),
        skip_s6=bool(args.skip_s6),
        seed=args.seed,
        repeats=args.repeats,
        no_retry=bool(args.no_retry),
        skip_completed=not bool(args.no_skip_completed),
        continue_on_failure=bool(args.continue_on_failure),
        engine_filter=_parse_csv_filter(args.engine_filter),
        scenario_filter=_parse_csv_filter(args.scenario_filter),
        template_filter=_parse_csv_filter(args.template_filter),
        max_runs=args.max_runs,
        deadline_hours=float(args.deadline_hours),
        adapter_timeout_s=float(args.adapter_timeout_s),
        poll_interval_s=float(args.poll_interval_s),
        verify_promotion=not bool(args.no_verify_promotion),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
