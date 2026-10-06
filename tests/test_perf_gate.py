from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.harness.perf_gate import check

BASELINE = yaml.safe_load((Path(__file__).resolve().parents[1] / "ci" / "perf_baseline.yaml").read_text())


def _ci(mean: float) -> dict:
    return {"mean": mean, "ci_low": mean, "ci_high": mean, "std": 0.0, "n": 2}


def _result(name: str, cells: list[tuple[dict, dict]], failed: bool = False) -> dict:
    return {
        "name": name,
        "trials": [{"trial_id": "t0", "status": "failed" if failed else "ok"}],
        "cells": [{"cell_id": str(i), "params": p, "metrics": {k: _ci(v) for k, v in m.items()}}
                  for i, (p, m) in enumerate(cells)],
    }


SIM_OK = {"errors": 0, "rejected": 0, "ok": 60, "ttft_p50_ms": 875, "tpot_p50_ms": 11.6, "slo_attainment": 1.0}
LLAMA_OK = {"errors": 0, "ok": 12, "output_tokens_per_s": 15, "ttft_p50_ms": 1500}


def _both(sim: dict = SIM_OK, llama: dict = LLAMA_OK, failed: bool = False) -> list[dict]:
    return [_result("ci-smoke-sim", [({}, sim)]),
            _result("ci-smoke-llamacpp", [({"workload.concurrency": 1}, llama), ({"workload.concurrency": 2}, llama)],
                    failed)]


def test_local_calibration_passes() -> None:
    assert check(BASELINE, _both()) == []


def test_all_errors_with_ok_trials_fails() -> None:
    problems = check(BASELINE, _both(sim={**SIM_OK, "errors": 60, "ok": 0}))
    assert any("errors 60 > max 0" in p for p in problems) and any("ok 0 < min 60" in p for p in problems)


def test_latency_drift_missing_metric_failed_trial_and_missing_run() -> None:
    assert any("ttft_p50_ms" in p for p in check(BASELINE, _both(sim={**SIM_OK, "ttft_p50_ms": 1100})))
    no_tps = {k: v for k, v in LLAMA_OK.items() if k != "output_tokens_per_s"}
    assert any("missing metric output_tokens_per_s" in p for p in check(BASELINE, _both(llama=no_tps)))
    assert any("failed trials" in p for p in check(BASELINE, _both(failed=True)))
    assert check(BASELINE, _both()[:1]) == ["ci-smoke-llamacpp: no result bundle"]


def test_cell_selector_scopes_bounds() -> None:
    slow = {**LLAMA_OK, "ttft_p50_ms": 20_000}
    results = [_both()[0], _result("ci-smoke-llamacpp", [({"workload.concurrency": 1}, LLAMA_OK),
                                                        ({"workload.concurrency": 2}, slow)])]
    assert check(BASELINE, results) == []  # the TTFT ceiling applies to concurrency 1 only
