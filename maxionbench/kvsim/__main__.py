"""Run a KV-cache simulation spec: matrix x repeats over the pinned AgentX traces, in parallel.

    python -m maxionbench.kvsim experiments/k1_agent_kv.yaml [--jobs 6] [--out results]

Writes results/<run_id>/results.json in the harness result format (cells with 95% CIs, provenance).
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import fields
from datetime import datetime, timezone
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from maxionbench.datasets.sources import verified_path
from maxionbench.harness.provenance import make_provenance
from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells
from maxionbench.kvsim.sim import SimParams, simulate
from maxionbench.kvsim.traces import TRACE_FILE, Session, load_sessions
from maxionbench.schemas.result_schema import utc_now_iso

SPEC_SCHEMA = "maxionbench-kvsim-v1"
_SESSIONS: list[Session] = []  # loaded once in the parent, shared with forked workers


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    allowed = {f.name for f in fields(SimParams)}
    unknown = (set(spec.get("params", {})) | set(spec.get("matrix", {}))) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown sim params {sorted(unknown)}")
    return spec


def plan(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any], int, int]]:
    """(cell_id, cell params, repeat, seed) for every trial."""
    axes = list(spec.get("matrix", {}).items())
    combos = [dict(zip([k for k, _ in axes], c)) for c in itertools.product(*[v for _, v in axes])] or [{}]
    trials = []
    for cell in combos:
        cell_id = "/".join(f"{k}={v}" for k, v in cell.items()) or "default"
        for rep in range(int(spec.get("repeats", 1))):
            trials.append((cell_id, cell, rep, int(spec.get("seed", 0)) * 1000 + rep))
    return trials


def _run(job: tuple[str, dict[str, Any], int, int, dict[str, Any]]) -> TrialResult:
    cell_id, cell, rep, seed, base = job
    params = SimParams(**{**base, **cell})
    started, t0 = utc_now_iso(), time.perf_counter()
    metrics = simulate(_SESSIONS, params, seed)
    return TrialResult(
        trial_id=f"{cell_id}/r{rep}", cell_id=cell_id, repeat=rep, seed=seed, status="ok", started_at=started,
        duration_s=round(time.perf_counter() - t0, 3), host_load_1m_before=0.0, quiet_host_ok=True,
        metrics={k: round(v, 6) for k, v in metrics.items()}, requests_per_endpoint=[],
        target={"kind": "kvsim", **{k: getattr(params, k) for k in (f.name for f in fields(SimParams))}}, error=None)


def run(spec_path: Path, jobs: int, out_root: Path) -> Path:
    global _SESSIONS
    spec = load_spec(spec_path)
    started_at = utc_now_iso()
    trace = verified_path(spec.get("trace", TRACE_FILE))
    _SESSIONS = load_sessions(trace, idle_cap_s=float(spec.get("idle_cap_s", 300.0)), limit=spec.get("sessions"))
    print(f"loaded {len(_SESSIONS)} sessions from {trace}", file=sys.stderr)
    trials = plan(spec)
    base = spec.get("params", {})
    with mp.get_context("fork").Pool(jobs) as pool:
        results = []
        for i, tr in enumerate(pool.imap_unordered(_run, [(*t, base) for t in trials]), 1):
            results.append(tr)
            m = tr.metrics
            print(f"[{i}/{len(trials)}] {tr.trial_id}: hit={m['token_hit_rate']:.4f} "
                  f"evicted={m['miss_evicted_share']:.4f} routing={m['miss_routing_share']:.4f} "
                  f"({tr.duration_s:.0f}s)", file=sys.stderr)
    order = {(cell_id, rep): i for i, (cell_id, _, rep, _) in enumerate(trials)}
    results.sort(key=lambda t: order[(t.cell_id, t.repeat)])
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    cells = {cell_id: cell for cell_id, cell, _, _ in trials}
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"], description=spec.get("description", ""),
        spec=spec, provenance=make_provenance(spec, started_at, {
            "trace": spec.get("trace", TRACE_FILE), "sessions_loaded": len(_SESSIONS),
            "trials_planned": len(trials), "trials_completed": len(results)}),
        trials=results, cells=aggregate_cells(results, cells))
    (out_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}", file=sys.stderr)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="python -m maxionbench.kvsim", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    run(args.spec, args.jobs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
