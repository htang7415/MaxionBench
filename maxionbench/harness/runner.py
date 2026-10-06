"""Execute planned trials and write a versioned, provenance-stamped result bundle.

Bundle layout: <out>/<run_id>/{spec.yaml, results.json, requests.jsonl, logs/}
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from maxionbench.harness.budget import BudgetLedger, ModelPrice, Reservation, cost_usd
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
from maxionbench.harness.workloads import Workload, make_workload, warmup_specs
from maxionbench.rag.llm_client import chat_completion
from maxionbench.rag.loadgen import RequestRecord, RequestSpec, run_closed_loop, run_open_loop, summarize
from maxionbench.runtime.system_info import collect_system_info
from maxionbench.schemas.result_schema import stable_config_fingerprint, utc_now_iso

TargetFactory = Callable[[str, dict[str, Any], Path], Target]
LedgerFactory = Callable[[float], BudgetLedger]


def run_experiment(
    spec: ExperimentSpec,
    out_root: Path,
    *,
    target_factory: TargetFactory = make_target,
    ledger_factory: LedgerFactory = BudgetLedger,
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
    pool = _TargetPool(target_factory, reuse=spec.reuse_targets)
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as req_fh:
        try:
            for n, trial in enumerate(trials, start=1):
                result = _run_trial(spec, trial, out_dir, pool, ledger_factory, req_fh, scrub, run_id)
                results.append(result)
                _write_result(out_dir, spec, spec_dict, run_id, started_at, key_present, trials, results)  # checkpoint
                headline = {k: result.metrics.get(k) for k in ("slo_attainment", "goodput_rps", "ttft_p50_ms")}
                log(f"[{n}/{len(trials)}] {trial.trial_id} {result.status} {headline} {result.error or ''}")
        finally:
            pool.close()

    result = _write_result(out_dir, spec, spec_dict, run_id, started_at, key_present, trials, results)
    return out_dir, result


def _run_trial(
    spec: ExperimentSpec,
    trial: Trial,
    out_dir: Path,
    pool: "_TargetPool",
    ledger_factory: LedgerFactory,
    req_fh: Any,
    scrub: Callable[[str], str],
    run_id: str,
) -> TrialResult:
    load_before, quiet_ok, foreign_engines = wait_for_quiet_host(spec.quiet_host)
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    target_desc: dict[str, Any] = {}
    try:
        workload = make_workload(spec.workload.kind, trial.workload_params, trial.seed)
        target, reused = pool.acquire(trial, out_dir / "logs" / trial.trial_id)
        if not reused and spec.quiet_host.min_available_gb > 0:
            free_gb = available_memory_gb()
            if free_gb < spec.quiet_host.min_available_gb:
                raise RuntimeError(
                    f"insufficient memory to start {trial.target_kind}: {free_gb:.1f} GB available "
                    f"< {spec.quiet_host.min_available_gb:.1f} GB required"
                )
        target_desc = {**target.describe(), "reused": reused, "foreign_engine_processes": foreign_engines}
        options = target.request_options()
        options["extra_body"] = {**options.get("extra_body", {}), **workload.extra_body}
        send = functools.partial(chat_completion, **options)
        pricing = target.pricing()
        ledger: BudgetLedger | None = None
        reservation: Reservation | None = None
        if pricing is not None and workload.warmup_requests:
            raise ValueError("warmup_requests is not allowed for paid targets (unbilled spend)")
        if pricing is not None:  # paid target: reserve the worst-case cost before any request
            _, price, cap = pricing
            ledger = ledger_factory(cap)
            reservation = ledger.reserve(estimate_cost_usd(workload, price), f"{run_id}/{trial.trial_id}")
        started_requests = False
        try:
            if not reused:
                pool.start(target, trial)
                for spec_ in warmup_specs(workload.warmup_requests):
                    send(target.base_urls[0], spec_.messages, max_tokens=8, timeout_s=workload.timeout_s)
            started_requests = True
            records, duration = _drive(workload, target, send, trial.seed)
            pool.release()
        except BaseException:
            pool.close()  # never reuse a target after a failed trial
            if ledger is not None and reservation is not None:
                if started_requests:  # requests may have been billed; charge the full estimate
                    ledger.commit(reservation, reservation.estimate_usd, {"note": "trial failed; estimate charged"})
                else:
                    ledger.release(reservation)
            raise
        if ledger is not None and reservation is not None and pricing is not None:
            actual, usage = actual_cost_usd(records, workload, pricing[1])
            ledger.commit(reservation, actual, usage)
            target_desc = {**target_desc, "spend_usd": round(actual, 6), "usage": usage}
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


def _write_result(
    out_dir: Path,
    spec: ExperimentSpec,
    spec_dict: dict[str, Any],
    run_id: str,
    started_at: str,
    key_present: bool,
    trials: list[Trial],
    results: list[TrialResult],
) -> ExperimentResult:
    """Build the result from the trials finished so far and replace results.json atomically,
    so an interrupted run keeps every completed trial."""
    done = {r.cell_id for r in results}
    cell_params = {t.cell_id: t.cell_params for t in trials if t.cell_id in done}
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
                "trials_planned": len(trials),
                "trials_completed": len(results),
            },
        ),
        trials=list(results),
        cells=aggregate_cells(results, cell_params),
    )
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_dir / "results.json")
    return result


class _TargetPool:
    """Starts targets for trials. With reuse, a started target stays up while consecutive trials
    share its kind and params (the planner groups those trials); otherwise every trial gets a fresh one."""

    def __init__(self, factory: TargetFactory, *, reuse: bool) -> None:
        self.factory = factory
        self.reuse = reuse
        self.current: Target | None = None
        self.key: str | None = None

    @staticmethod
    def _key(trial: Trial) -> str:
        return trial.target_kind + json.dumps(trial.target_params, sort_keys=True, default=str)

    def acquire(self, trial: Trial, log_dir: Path) -> tuple[Target, bool]:
        if self.reuse and self.current is not None and self.key == self._key(trial):
            return self.current, True
        self.close()
        return self.factory(trial.target_kind, trial.target_params, log_dir), False

    def start(self, target: Target, trial: Trial) -> None:
        target.__enter__()
        self.current, self.key = target, self._key(trial)

    def release(self) -> None:
        if not self.reuse:
            self.close()

    def close(self) -> None:
        if self.current is not None:
            current, self.current, self.key = self.current, None, None
            current.__exit__(None, None, None)


def _drive(
    workload: Workload, target: Target, send: Callable[..., Any], seed: int
) -> tuple[list[RequestRecord], float]:
    if workload.concurrency is not None:
        return run_closed_loop(
            workload.specs, base_urls=target.base_urls, picker=target.picker(),
            concurrency=workload.concurrency, timeout_s=workload.timeout_s,
            max_tokens=workload.max_tokens, send=send,
        )
    assert workload.rate_rps is not None
    return run_open_loop(
        workload.specs, base_urls=target.base_urls, picker=target.picker(), rate_rps=workload.rate_rps,
        max_in_flight=workload.max_in_flight, timeout_s=workload.timeout_s,
        max_tokens=workload.max_tokens, seed=seed, send=send,
    )


def estimated_tokens(spec: RequestSpec) -> int:
    """Conservative prompt-token estimate (~3 chars/token plus per-message overhead)."""
    return sum(len(m.get("content", "")) // 3 + 8 for m in spec.messages)


def estimate_cost_usd(workload: Workload, price: ModelPrice) -> float:
    prompt = sum(estimated_tokens(s) for s in workload.specs)
    return cost_usd(price, input_tokens=prompt, output_tokens=workload.max_tokens * len(workload.specs))


def actual_cost_usd(records: list[RequestRecord], workload: Workload, price: ModelPrice) -> tuple[float, dict[str, int]]:
    """Bill from provider-reported usage; requests without usage are charged their estimate."""
    by_id = {s.request_id: s for s in workload.specs}
    usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "estimated_requests": 0}
    for r in records:
        if r.status == "ok" and r.prompt_tokens > 0:
            usage["input_tokens"] += r.prompt_tokens
            usage["cached_tokens"] += r.cached_tokens
            usage["output_tokens"] += r.completion_tokens
        elif r.status not in ("rejected", "no_endpoint"):  # may have been billed without usage data
            usage["input_tokens"] += estimated_tokens(by_id[r.request_id])
            usage["output_tokens"] += workload.max_tokens
            usage["estimated_requests"] += 1
    cost = cost_usd(price, input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                    cached_tokens=usage["cached_tokens"])
    return cost, usage


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
        "requests_per_s": float(summary["requests_per_s"]),
        "output_tokens_per_s": float(summary["output_tokens_per_s"]),
    }
    for group in ("ttft", "e2e", "tpot"):
        for key, value in (summary.get(group) or {}).items():
            metrics[f"{group}_{key}"] = float(value)
    return metrics


_PY_ENGINE_MODULES = ("vllm.entrypoints", "mlx_lm.server", "mlx_lm")


def is_engine_command(command: str) -> bool:
    """True if the *executable being run* is an inference server; mere mentions in a shell line do not count."""
    tokens = command.split()
    if not tokens:
        return False
    exe = tokens[0].rsplit("/", 1)[-1]
    if exe == "llama-server":
        return True
    if exe == "vllm":
        return len(tokens) > 1 and tokens[1] == "serve"
    if exe.lower().startswith("python"):
        rest = tokens[1:]
        if len(rest) >= 2 and rest[0].rsplit("/", 1)[-1] == "vllm" and rest[1] == "serve":
            return True  # e.g. "Python /venv/bin/vllm serve model"
        if "-m" in rest:
            i = rest.index("-m")
            module = rest[i + 1] if i + 1 < len(rest) else ""
            return module.startswith(_PY_ENGINE_MODULES) and (module != "mlx_lm" or "server" in rest[i + 2 : i + 3])
    return False


def foreign_engine_pids(ps_output: str | None = None, self_pid: int | None = None) -> list[int]:
    """Inference-engine processes not started by this harness (outside our own process tree).

    Load average does not reflect GPU contention, so another engine on the host must be detected directly.
    """
    if ps_output is None:
        ps_output = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    me = os.getpid() if self_pid is None else self_pid
    procs: list[tuple[int, int, str]] = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs.append((int(parts[0]), int(parts[1]), parts[2]))
    children: dict[int, list[int]] = {}
    for pid, ppid, _ in procs:
        children.setdefault(ppid, []).append(pid)
    own, stack = {me}, [me]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in own:
                own.add(child)
                stack.append(child)
    return [pid for pid, _, cmd in procs if pid not in own and is_engine_command(cmd)]


def available_memory_gb(vm_stat: str | None = None) -> float:
    """Reclaimable memory on macOS: free + inactive + speculative + purgeable pages (from `vm_stat`)."""
    if vm_stat is None:
        try:
            vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return float("inf")  # unknown platform: do not block
    page = 16384
    pages: dict[str, int] = {}
    for line in vm_stat.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        name, _, value = line.partition(":")
        if value.strip().rstrip(".").isdigit():
            pages[name.strip()] = int(value.strip().rstrip("."))
    keys = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    if not any(k in pages for k in keys):
        return float("inf")
    return sum(pages.get(k, 0) for k in keys) * page / 1024**3


ENGINE_IMAGE_HINTS = ("vllm", "llama", "sglang", "inference-sim", "text-generation-inference", "tgi", "mlx", "ollama")


def foreign_engine_containers(docker_ps: str | None = None) -> list[str]:
    """Running inference-server containers not started by this harness (names prefixed `maxionbench-`).

    Containerized engines are invisible to `ps` on macOS (they show up only as the Docker VM process).
    """
    if docker_ps is None:
        try:
            out = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return []
        docker_ps = out.stdout if out.returncode == 0 else ""
    found = []
    for line in docker_ps.splitlines():
        name, _, image = line.partition("\t")
        if name.startswith("maxionbench-") or not image:
            continue
        if any(hint in image.lower() for hint in ENGINE_IMAGE_HINTS):
            found.append(name)
    return found


def foreign_engines() -> list[str]:
    return [f"pid:{p}" for p in foreign_engine_pids()] + [f"container:{c}" for c in foreign_engine_containers()]


def wait_for_quiet_host(
    policy: QuietHost, *, poll_s: float = 5.0, foreign: Callable[[], list[Any]] = foreign_engines
) -> tuple[float, bool, int]:
    """Wait (bounded) until load is under the threshold and no foreign inference engine is running.

    Returns (1-minute load, quiet, number of foreign engine processes)."""
    deadline = time.monotonic() + policy.wait_s
    while True:
        load = round(os.getloadavg()[0], 2)
        others = len(foreign())
        if load <= policy.max_load_1m and others == 0:
            return load, True, 0
        if time.monotonic() >= deadline:
            return load, False, others
        time.sleep(poll_s)


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()
