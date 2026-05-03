from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_promotion_gate import REQUIRED_ADAPTERS, verify_portable_promotion_gate, verify_promotion_gate


def _write_matrix(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["adapter", "status"])
        writer.writeheader()
        for adapter, status in rows:
            writer.writerow({"adapter": adapter, "status": status})


def _write_readiness_summary(path: Path, *, pass_value: bool = True) -> None:
    payload = {
        "pass": pass_value,
        "required_adapters": list(REQUIRED_ADAPTERS),
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0 if pass_value else 1,
        "errors": [] if pass_value else [{"message": "adapter `qdrant` failed conformance"}],
        "conformance_rows": len(REQUIRED_ADAPTERS),
        "conformance_status_counts": {"pass": len(REQUIRED_ADAPTERS)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_portable_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _portable_row(
    *,
    scenario: str,
    budget: str,
    quality: float,
    engine: str = "mock",
    freshness_hit_at_5s: float | None = None,
    evidence_coverage_at_10: float | None = None,
    task_cost_est: float = 1.0,
    errors: int = 0,
    measure_requests: int = 100,
) -> dict:
    payload = {
        "profile": "maxionbench",
        "budget_level": budget,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "primary_quality_metric": "evidence_coverage@10" if scenario == "s3_multi_hop" else "ndcg_at_10",
        "primary_quality_value": quality,
        "task_cost_est": task_cost_est,
    }
    if freshness_hit_at_5s is not None:
        payload["freshness_hit_at_5s"] = freshness_hit_at_5s
    if evidence_coverage_at_10 is not None:
        payload["evidence_coverage_at_10"] = evidence_coverage_at_10
    return {
        "run_id": f"{budget}-{scenario}-{engine}-{task_cost_est}",
        "scenario": scenario,
        "engine": engine,
        "engine_version": "test",
        "dataset_bundle": "HOTPOT_PORTABLE" if scenario == "s3_multi_hop" else "D4",
        "dataset_hash": "fixture",
        "seed": 42,
        "repeat_idx": 0,
        "clients_read": 1,
        "clients_write": 2 if scenario == "s2_streaming_memory" else 0,
        "quality_target": 0.25,
        "search_params_json": json.dumps(payload, sort_keys=True),
        "ndcg_at_10": quality,
        "p50_ms": 1.0,
        "p95_ms": 2.0,
        "p99_ms": 3.0,
        "qps": 100.0,
        "errors": errors,
        "measure_requests": measure_requests,
    }


def _passing_b0_rows() -> list[dict]:
    return [
        _portable_row(scenario="s1_single_hop", budget="b0", quality=0.19, task_cost_est=0.3),
        _portable_row(
            scenario="s2_streaming_memory",
            budget="b0",
            quality=0.19,
            freshness_hit_at_5s=0.7,
            task_cost_est=0.2,
        ),
        _portable_row(
            scenario="s3_multi_hop",
            budget="b0",
            quality=0.23,
            evidence_coverage_at_10=0.23,
            task_cost_est=0.1,
        ),
    ]


def test_verify_promotion_gate_passes_for_portable_readiness_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    _write_readiness_summary(summary_path)

    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)

    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_checks_matrix_consistency(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    _write_readiness_summary(summary_path)
    _write_matrix(matrix_path, [(adapter, "pass") for adapter in REQUIRED_ADAPTERS] + [("mock", "pass")])

    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )

    assert summary["pass"] is False
    assert summary["matrix_observed"]["rows"] == len(REQUIRED_ADAPTERS) + 1


def test_verify_promotion_gate_fails_for_nonpassing_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    _write_readiness_summary(summary_path, pass_value=False)

    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)

    assert summary["pass"] is False
    assert "strict readiness summary reports pass=false" in summary["reasons"]


def test_verify_portable_promotion_gate_passes_b0_to_b1_and_writes_candidates(tmp_path: Path) -> None:
    results_path = tmp_path / "results.parquet"
    candidates_path = tmp_path / "promotion_candidates.json"
    _write_portable_results(results_path, _passing_b0_rows())

    summary = verify_portable_promotion_gate(
        results_path=results_path,
        from_budget="b0",
        out_candidates_path=candidates_path,
    )

    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["to_budget"] == "b1"
    assert summary["promoted_survivor_count"] == 3
    assert candidates_path.exists()


def test_verify_portable_promotion_gate_rejects_stale_s2_freshness(tmp_path: Path) -> None:
    results_path = tmp_path / "results.parquet"
    rows = _passing_b0_rows()
    rows[1] = _portable_row(
        scenario="s2_streaming_memory",
        budget="b0",
        quality=0.19,
        freshness_hit_at_5s=0.5,
    )
    _write_portable_results(results_path, rows)

    summary = verify_portable_promotion_gate(results_path=results_path, from_budget="b0")

    assert summary["pass"] is False
    assert summary["missing_survivors"] == ["s2_streaming_memory"]
    assert any("freshness_hit_at_5s" in reason for item in summary["rejections"] for reason in item["reasons"])


def test_verify_portable_promotion_gate_prunes_b1_survivors_per_scenario(tmp_path: Path) -> None:
    results_path = tmp_path / "results.parquet"
    rows = [
        _portable_row(scenario="s1_single_hop", budget="b1", quality=0.23, engine=f"engine-{idx}", task_cost_est=float(idx))
        for idx in range(4)
    ]
    rows.extend(
        [
            _portable_row(scenario="s2_streaming_memory", budget="b1", quality=0.23, freshness_hit_at_5s=0.9),
            _portable_row(
                scenario="s3_multi_hop",
                budget="b1",
                quality=0.28,
                evidence_coverage_at_10=0.28,
            ),
        ]
    )
    _write_portable_results(results_path, rows)

    summary = verify_portable_promotion_gate(results_path=results_path, from_budget="b1")

    assert summary["pass"] is True
    s1_survivors = [row for row in summary["survivors"] if row["scenario"] == "s1_single_hop"]
    assert len(s1_survivors) == 3
    assert [row["engine"] for row in s1_survivors] == ["engine-0", "engine-1", "engine-2"]


def test_verify_portable_promotion_gate_cli_dispatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_path = tmp_path / "results.parquet"
    _write_portable_results(results_path, _passing_b0_rows())

    code = cli_main(
        [
            "verify-promotion-gate",
            "--maxionbench-results",
            str(results_path),
            "--from-budget",
            "b0",
            "--json",
        ]
    )

    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["mode"] == "maxionbench-promotion"
    assert parsed["pass"] is True
