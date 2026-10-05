"""Execute planned trials and write a versioned, provenance-stamped result bundle.

Bundle layout: <out>/<run_id>/{spec.yaml, results.json, requests.jsonl, logs/}
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from maxionbench.harness.planner import Trial, plan
from maxionbench.harness.results import (
    RESULT_SCHEMA_VERSION,
    ExperimentResult,
    Provenance,
    TrialResult,
    aggregate_cells,
)
from maxionbench.harness.secrets import MissingSecretError, load_gemini_key, redact
from maxionbench.harness.spec import ExperimentSpec, QuietHost
from maxionbench.harness.targets import Target, make_target
from maxionbench.harness.workloads import make_workload
from maxionbench.rag.loadgen import run_open_loop, summarize
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.schemas.result_schema import stable_config_fingerprint, utc_now_iso

TargetFactory = Callable[[str, dict[str, Any], Path], Target]


def run_experiment(
    spec: ExperimentSpec,
    out_root: Path,
    *,
    target_factory: TargetFactory = make_target,
    log: Callable[[str], None] = lambda msg: print(f"[harness] {msg}", file=sys.stderr, flush=True),
) -> tuple[Path, ExperimentResult]:
    started_at = utc_now_iso()
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec.name}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    spec_dict = spec.to_dict()
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec_dict, sort_keys=False), encoding="utf-8")
    trials = plan(spec)
    log(f"{run_id}: {len(trials)} trials ({spec.repeats} repeats)")

    scrub, key_present = _scrubber()
    results: list[TrialResult] = []
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as req_fh:
        for n, trial in enumerate(trials, start=1):
            result = _run_trial(spec, trial, out_dir, target_factory, req_fh, scrub)
            results.append(result)
            headline = {k: result.metrics.get(k) for k in ("slo_attainment", "goodput_rps", "ttft_p50_ms")}
            log(f"[{n}/{len(trials)}] {trial.trial_id} {result.status} {headline} {result.error or ''}")

    cell_params = {t.cell_id: t.cell_params for t in trials}
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=run_id,
        name=spec.name,
        description=spec.description,
        spec=spec_dict,
        provenance=Provenance(
            git_commit=_git(["rev-parse", "HEAD"]) or "unknown",
            # Untracked source counts: results from uncommitted code are not reproducible from git_commit.
            git_dirty=bool(_git(["status", "--porcelain"])),
            spec_fingerprint=stable_config_fingerprint(spec_dict),
            started_at=started_at,
            finished_at=utc_now_iso(),
            host=collect_system_info(),
            tools={
                "python": platform.python_version(),
                "harness_result_schema": RESULT_SCHEMA_VERSION,
                "gemini_key_present": key_present,  # presence only, never the value
            },
        ),
        trials=results,
        cells=aggregate_cells(results, cell_params),
    )
    (out_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out_dir, result


def _run_trial(
    spec: ExperimentSpec,
    trial: Trial,
    out_dir: Path,
    target_factory: TargetFactory,
    req_fh: Any,
    scrub: Callable[[str], str],
) -> TrialResult:
    load_before, quiet_ok = wait_for_quiet_host(spec.quiet_host)
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    target_desc: dict[str, Any] = {}
    try:
        workload = make_workload(spec.workload.kind, trial.workload_params, trial.seed)
        target = target_factory(spec.target.kind, trial.target_params, out_dir / "logs" / trial.trial_id)
        target_desc = target.describe()
        with target:
            records, duration = run_open_loop(
                workload.specs,
                base_urls=target.base_urls,
                picker=target.picker(),
                rate_rps=workload.rate_rps,
                max_in_flight=workload.max_in_flight,
                timeout_s=workload.timeout_s,
                max_tokens=workload.max_tokens,
                seed=trial.seed,
            )
        summary = summarize(records, duration_s=duration, ttft_slo_s=spec.slo.ttft_s, e2e_slo_s=spec.slo.e2e_s)
        for r in records:
            row = {"trial_id": trial.trial_id, "cell_id": trial.cell_id, **asdict(r)}
            req_fh.write(scrub(json.dumps(row)) + "\n")
        return TrialResult(
            trial_id=trial.trial_id,
            cell_id=trial.cell_id,
            repeat=trial.repeat,
            seed=trial.seed,
            status="ok",
            started_at=started_at,
            duration_s=round(time.perf_counter() - t0, 3),
            host_load_1m_before=load_before,
            quiet_host_ok=quiet_ok,
            metrics=flatten_summary(summary),
            requests_per_endpoint=[sum(1 for r in records if r.endpoint == i) for i in range(len(target.base_urls))],
            target=target_desc,
            error=None,
        )
    except Exception as exc:  # a failed trial is recorded, never silently dropped
        return TrialResult(
            trial_id=trial.trial_id,
            cell_id=trial.cell_id,
            repeat=trial.repeat,
            seed=trial.seed,
            status="failed",
            started_at=started_at,
            duration_s=round(time.perf_counter() - t0, 3),
            host_load_1m_before=load_before,
            quiet_host_ok=quiet_ok,
            metrics={},
            requests_per_endpoint=[],
            target=target_desc,
            error=scrub(f"{type(exc).__name__}: {exc}"),
        )


def _scrubber() -> tuple[Callable[[str], str], bool]:
    """Redact any configured API key from text bound for logs or result bundles."""
    try:
        key = load_gemini_key()
    except MissingSecretError:
        return (lambda text: text), False
    return (lambda text: redact(text, key)), True


def flatten_summary(summary: dict[str, Any]) -> dict[str, float]:
    counts = summary["status_counts"]
    metrics: dict[str, float] = {
        "requests": float(summary["requests"]),
        "ok": float(counts.get("ok", 0)),
        "rejected": float(counts.get("rejected", 0)),
        "errors": float(sum(v for k, v in counts.items() if k not in ("ok", "rejected"))),
        "degraded": float(summary["degraded"]),
        "slo_attainment": float(summary["slo_attainment"]),
        "goodput_rps": float(summary["goodput_rps"]),
        "prefix_cache_hit_ratio": float(summary["prefix_cache_hit_ratio"]),
        "duration_s": float(summary["duration_s"]),
    }
    for group in ("ttft", "e2e"):
        for key, value in (summary.get(group) or {}).items():
            metrics[f"{group}_{key}"] = float(value)
    return metrics


def wait_for_quiet_host(policy: QuietHost, *, poll_s: float = 5.0) -> tuple[float, bool]:
    """Wait (bounded) for the 1-minute load average to fall below the threshold."""
    deadline = time.monotonic() + policy.wait_s
    while True:
        load = round(os.getloadavg()[0], 2)
        if load <= policy.max_load_1m:
            return load, True
        if time.monotonic() >= deadline:
            return load, False
        time.sleep(poll_s)


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()
