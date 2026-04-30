"""Execute a generated run matrix sequentially from the terminal."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import time
from typing import Any, Iterable

from maxionbench.orchestration.config_schema import load_run_config
from maxionbench.orchestration.run_matrix import RunMatrixRow, load_run_matrix
from maxionbench.orchestration.runner import run_from_config
from maxionbench.schemas.result_schema import read_run_status
from maxionbench.tools.wait_adapter import wait_for_adapter

_SERVICE_ENGINES = {"qdrant", "pgvector"}


def execute_run_matrix(
    *,
    matrix_path: Path,
    lane: str = "cpu",
    budget: str | None = None,
    seed: int | None = None,
    repeats: int | None = None,
    no_retry: bool = False,
    skip_completed: bool = False,
    continue_on_failure: bool = False,
    engine_filter: set[str] | None = None,
    scenario_filter: set[str] | None = None,
    template_filter: set[str] | None = None,
    max_runs: int | None = None,
    deadline_hours: float | None = None,
    adapter_timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    matrix = load_run_matrix(matrix_path)
    effective_budget = budget if budget is not None else matrix.budget_level
    selected_rows = list(
        _filter_rows(
            rows=matrix.iter_rows(lane=lane),
            engine_filter=engine_filter,
            scenario_filter=scenario_filter,
            template_filter=template_filter,
            max_runs=max_runs,
        )
    )
    summary: dict[str, Any] = {
        "matrix_path": str(matrix_path.expanduser().resolve()),
        "lane": lane,
        "budget": effective_budget,
        "selected_rows": len(selected_rows),
        "completed_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "failures": [],
        "warnings": [],
    }
    overrides: dict[str, Any] = {}
    if effective_budget is not None:
        overrides["budget_level"] = effective_budget
    if seed is not None:
        overrides["seed"] = seed
    if repeats is not None:
        overrides["repeats"] = repeats
    if no_retry:
        overrides["no_retry"] = True

    for row in selected_rows:
        config_path = Path(row.config_path).expanduser().resolve()
        cfg = load_run_config(config_path)
        output_dir = Path(cfg.output_dir).expanduser().resolve()
        if skip_completed and _run_status_is_success(output_dir):
            summary["skipped_rows"] = int(summary["skipped_rows"]) + 1
            continue
        if _requires_wait(cfg.engine, cfg.adapter_options):
            wait_for_adapter(
                adapter_name=cfg.engine,
                adapter_options=dict(cfg.adapter_options),
                timeout_s=adapter_timeout_s,
                poll_interval_s=poll_interval_s,
            )
        try:
            run_from_config(config_path, cli_overrides=overrides or None)
            summary["completed_rows"] = int(summary["completed_rows"]) + 1
            _append_deadline_warning(summary=summary, started=started, deadline_hours=deadline_hours)
        except Exception as exc:
            failure = {
                "config_path": str(config_path),
                "engine": cfg.engine,
                "scenario": cfg.scenario,
                "template_name": row.template_name,
                "error": str(exc),
            }
            cast_failures = list(summary["failures"])
            cast_failures.append(failure)
            summary["failures"] = cast_failures
            summary["failed_rows"] = int(summary["failed_rows"]) + 1
            if not continue_on_failure:
                raise RuntimeError(json.dumps(summary, sort_keys=True)) from exc
    return summary


def _append_deadline_warning(*, summary: dict[str, Any], started: float, deadline_hours: float | None) -> None:
    if deadline_hours is None or deadline_hours <= 0:
        return
    completed = int(summary.get("completed_rows", 0))
    skipped = int(summary.get("skipped_rows", 0))
    selected = int(summary.get("selected_rows", 0))
    observed = completed + skipped
    if observed < 1 or selected < 1:
        return
    elapsed_s = time.perf_counter() - started
    average_s = elapsed_s / max(observed, 1)
    projected_s = average_s * selected
    deadline_s = float(deadline_hours) * 3600.0
    if projected_s <= deadline_s:
        return
    warnings = list(summary.get("warnings", []))
    if warnings and "projected runtime" in str(warnings[-1].get("message", "")):
        return
    warnings.append(
        {
            "type": "deadline_projection",
            "message": (
                f"projected runtime {projected_s / 3600.0:.2f}h exceeds "
                f"deadline {float(deadline_hours):.2f}h"
            ),
            "elapsed_s": elapsed_s,
            "projected_s": projected_s,
            "deadline_s": deadline_s,
            "observed_rows": observed,
            "selected_rows": selected,
        }
    )
    summary["warnings"] = warnings


def _filter_rows(
    *,
    rows: Iterable[RunMatrixRow],
    engine_filter: set[str] | None,
    scenario_filter: set[str] | None,
    template_filter: set[str] | None,
    max_runs: int | None,
) -> Iterable[RunMatrixRow]:
    count = 0
    for row in rows:
        if engine_filter and str(row.engine) not in engine_filter:
            continue
        if scenario_filter and str(row.scenario) not in scenario_filter:
            continue
        if template_filter and str(Path(row.template_name).stem) not in template_filter and str(row.template_name) not in template_filter:
            continue
        yield row
        count += 1
        if max_runs is not None and count >= max_runs:
            break


def _run_status_is_success(output_dir: Path) -> bool:
    status_path = output_dir / "run_status.json"
    if not status_path.exists():
        return False
    payload = read_run_status(status_path)
    return str(payload.get("status", "")).strip().lower() == "success"


def _requires_wait(engine: str, adapter_options: dict[str, Any]) -> bool:
    normalized = str(engine).strip().lower()
    if normalized in _SERVICE_ENGINES:
        return True
    if normalized == "lancedb-service":
        return not bool(str(adapter_options.get("inproc_uri", "")).strip())
    return False


def _parse_csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return items or None


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description="Execute a generated run matrix sequentially.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--lane", default="cpu", choices=["cpu", "gpu", "all"])
    parser.add_argument("--budget", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--engine-filter", default=None)
    parser.add_argument("--scenario-filter", default=None)
    parser.add_argument("--template-filter", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--deadline-hours", type=float, default=None)
    parser.add_argument("--adapter-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = execute_run_matrix(
        matrix_path=Path(args.matrix),
        lane=str(args.lane),
        budget=args.budget,
        seed=args.seed,
        repeats=args.repeats,
        no_retry=bool(args.no_retry),
        skip_completed=bool(args.skip_completed),
        continue_on_failure=bool(args.continue_on_failure),
        engine_filter=_parse_csv_filter(args.engine_filter),
        scenario_filter=_parse_csv_filter(args.scenario_filter),
        template_filter=_parse_csv_filter(args.template_filter),
        max_runs=args.max_runs,
        deadline_hours=args.deadline_hours,
        adapter_timeout_s=float(args.adapter_timeout_s),
        poll_interval_s=float(args.poll_interval_s),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
