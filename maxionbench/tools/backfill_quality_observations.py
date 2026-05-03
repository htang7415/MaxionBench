"""Backfill per-query quality observations for archived MaxionBench runs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from maxionbench.orchestration.config_schema import RunConfig, expand_env_placeholders
from maxionbench.orchestration.runner import (
    _create_benchmark_adapter,
    _load_portable_s1_dataset,
    _load_portable_s2_datasets,
    _load_portable_s3_dataset,
)
from maxionbench.scenarios.portable_text_retrieval import PortableTextConfig, evaluate_text_queries, ingest_text_dataset

_PORTABLE_SCENARIOS = {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}


def backfill_quality_observations(
    *,
    input_root: Path,
    engine_filter: set[str] | None = None,
    embedding_filter: set[str] | None = None,
    scenario_filter: set[str] | None = None,
) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in input_root.expanduser().resolve().rglob("results.parquet"))
    summary: dict[str, Any] = {
        "input_root": str(input_root.expanduser().resolve()),
        "run_dirs_checked": len(run_dirs),
        "rows_backfilled": 0,
        "observation_files": [],
        "skipped_rows": 0,
    }
    for run_dir in run_dirs:
        results_path = run_dir / "results.parquet"
        config_path = run_dir / "config_resolved.yaml"
        if not config_path.exists():
            continue
        cfg = _load_resolved_run_config(config_path)
        if str(cfg.scenario) not in _PORTABLE_SCENARIOS:
            continue
        if engine_filter and str(cfg.engine) not in engine_filter:
            continue
        if embedding_filter and str(cfg.embedding_model) not in embedding_filter:
            continue
        if scenario_filter and str(cfg.scenario) not in scenario_filter:
            continue

        frame = pd.read_parquet(results_path)
        if frame.empty:
            continue
        updated = False
        for idx, row in frame.iterrows():
            payload = _payload_from_row(row)
            if str(payload.get("observation_path") or "").strip():
                summary["skipped_rows"] = int(summary["skipped_rows"]) + 1
                continue
            search_params = payload.get("search_params")
            if not isinstance(search_params, Mapping):
                search_params = {}
            observations = _evaluate_quality_observations(
                cfg=cfg,
                config_path=config_path,
                search_params=dict(search_params),
                clients_read=int(row.get("clients_read", cfg.clients_read)),
            )
            observation_path = run_dir / "logs" / "observations" / f"backfill_{row['run_id']}.jsonl"
            _write_jsonl(observation_path, observations)
            payload["observation_path"] = str(observation_path)
            frame.at[idx, "search_params_json"] = json.dumps(payload, sort_keys=True)
            updated = True
            summary["rows_backfilled"] = int(summary["rows_backfilled"]) + 1
            observation_files = list(summary["observation_files"])
            observation_files.append(str(observation_path))
            summary["observation_files"] = observation_files
        if updated:
            frame.to_parquet(results_path, index=False)
    return summary


def _evaluate_quality_observations(
    *,
    cfg: Any,
    config_path: Path,
    search_params: dict[str, Any],
    clients_read: int,
) -> list[dict[str, Any]]:
    if cfg.scenario == "s1_single_hop":
        dataset = _load_portable_s1_dataset(cfg, config_path=config_path)
    elif cfg.scenario == "s2_streaming_memory":
        dataset, _ = _load_portable_s2_datasets(cfg, config_path=config_path)
    elif cfg.scenario == "s3_multi_hop":
        dataset = _load_portable_s3_dataset(cfg, config_path=config_path)
    else:
        raise ValueError(f"unsupported scenario: {cfg.scenario}")

    adapter = _create_benchmark_adapter(cfg=cfg)
    observations: list[dict[str, Any]] = []
    try:
        ingest_text_dataset(adapter, dataset)
        evaluate_text_queries(
            adapter=adapter,
            cfg=PortableTextConfig(
                top_k=cfg.top_k,
                clients_read=max(1, int(clients_read)),
                sla_threshold_ms=cfg.sla_threshold_ms,
                warmup_s=0.0,
                steady_state_s=1.0,
                phase_timing_mode="bounded",
                phase_max_requests_per_phase=None,
                search_params=search_params,
            ),
            dataset=dataset,
            observation_sink=observations.append,
        )
    finally:
        try:
            adapter.drop(collection="maxionbench")
        except Exception:
            pass
    return observations


def _payload_from_row(row: pd.Series) -> dict[str, Any]:
    raw = row.get("search_params_json")
    if isinstance(raw, str) and raw.strip():
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    return {}


def _load_resolved_run_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    allowed = {field.name for field in fields(RunConfig)}
    filtered = {key: value for key, value in expand_env_placeholders(payload).items() if key in allowed}
    return RunConfig(**filtered)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _parse_csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def parse_args(argv: list[str] | None = None) -> Any:
    parser = ArgumentParser(description="Backfill portable per-query quality observation JSONL files.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--engine-filter", default=None)
    parser.add_argument("--embedding-filter", default=None)
    parser.add_argument("--scenario-filter", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = backfill_quality_observations(
        input_root=Path(args.input_root),
        engine_filter=_parse_csv_filter(args.engine_filter),
        embedding_filter=_parse_csv_filter(args.embedding_filter),
        scenario_filter=_parse_csv_filter(args.scenario_filter),
    )
    print(json.dumps(summary, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
