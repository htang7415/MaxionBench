from __future__ import annotations

import json

import pandas as pd
import pytest

from maxionbench.reports.tables import _table_t3_robustness


def _find_row(table: pd.DataFrame, *, scenario: str) -> pd.Series:
    row = table[table["scenario"] == scenario]
    assert len(row) == 1
    return row.iloc[0]


def test_t3_robustness_uses_recorded_s3_inflation_without_s1_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s3_churn_smooth",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s3-1",
                "p99_ms": 120.0,
                "sla_violation_rate": 0.01,
                "errors": 0,
                "search_params_json": json.dumps(
                    {
                        "s1_baseline_missing": False,
                        "p99_inflation_vs_s1_baseline": 3.0,
                    },
                    sort_keys=True,
                ),
            }
        ]
    )
    table = _table_t3_robustness(frame)
    row = _find_row(table, scenario="s3_churn_smooth")
    assert float(row["p99_inflation_median"]) == pytest.approx(3.0)
    assert int(row["p99_inflation_valid_rows"]) == 1
    assert int(row["p99_inflation_nan_rows"]) == 0
    assert str(row["p99_inflation_status"]) == "computed_all_rows"


def test_t3_robustness_s3_missing_recorded_baseline_returns_nan() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s3b_churn_bursty",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s3b-1",
                "p99_ms": 140.0,
                "sla_violation_rate": 0.03,
                "errors": 0,
                "search_params_json": json.dumps(
                    {
                        "s1_baseline_missing": True,
                        "s1_baseline_error": "missing matched S1 baseline",
                        "p99_inflation_vs_s1_baseline": None,
                    },
                    sort_keys=True,
                ),
            }
        ]
    )
    table = _table_t3_robustness(frame)
    row = _find_row(table, scenario="s3b_churn_bursty")
    assert pd.isna(row["p99_inflation_median"])
    assert int(row["p99_inflation_valid_rows"]) == 0
    assert int(row["p99_inflation_nan_rows"]) == 1
    assert str(row["p99_inflation_status"]) == "not_computable"


def test_t3_robustness_s3_legacy_fallback_recomputes_from_s1_when_record_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s1_ann_frontier",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s1-1",
                "p99_ms": 40.0,
                "sla_violation_rate": 0.0,
                "errors": 0,
                "search_params_json": "{}",
            },
            {
                "scenario": "s3_churn_smooth",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s3-1",
                "p99_ms": 120.0,
                "sla_violation_rate": 0.01,
                "errors": 0,
                "search_params_json": "{}",
            },
        ]
    )
    table = _table_t3_robustness(frame)
    row = _find_row(table, scenario="s3_churn_smooth")
    assert float(row["p99_inflation_median"]) == pytest.approx(3.0)
    assert int(row["p99_inflation_valid_rows"]) == 1
    assert int(row["p99_inflation_nan_rows"]) == 0
    assert str(row["p99_inflation_status"]) == "computed_all_rows"


def test_t3_robustness_marks_partial_when_some_rows_missing_inflation() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "s3_churn_smooth",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s3-1",
                "p99_ms": 120.0,
                "sla_violation_rate": 0.01,
                "errors": 0,
                "search_params_json": json.dumps(
                    {
                        "s1_baseline_missing": False,
                        "p99_inflation_vs_s1_baseline": 3.0,
                    },
                    sort_keys=True,
                ),
            },
            {
                "scenario": "s3_churn_smooth",
                "dataset_bundle": "D3",
                "dataset_hash": "synthetic-d3-v1",
                "engine": "mock",
                "clients_read": 32,
                "run_id": "run-s3-2",
                "p99_ms": 130.0,
                "sla_violation_rate": 0.02,
                "errors": 0,
                "search_params_json": json.dumps(
                    {
                        "s1_baseline_missing": True,
                        "s1_baseline_error": "missing matched S1 baseline",
                        "p99_inflation_vs_s1_baseline": None,
                    },
                    sort_keys=True,
                ),
            },
        ]
    )
    table = _table_t3_robustness(frame)
    row = _find_row(table, scenario="s3_churn_smooth")
    assert float(row["p99_inflation_median"]) == pytest.approx(3.0)
    assert int(row["p99_inflation_valid_rows"]) == 1
    assert int(row["p99_inflation_nan_rows"]) == 1
    assert str(row["p99_inflation_status"]) == "computed_partial_rows"
