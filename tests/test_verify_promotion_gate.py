from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import pandas as pd

from maxionbench.cli import main as cli_main
from maxionbench.tools.verify_promotion_gate import verify_portable_promotion_gate, verify_promotion_gate

STRICT_REQUIRED_ADAPTERS = [
    "qdrant",
    "milvus",
    "weaviate",
    "opensearch",
    "pgvector",
    "lancedb-service",
    "lancedb-inproc",
    "faiss-cpu",
    "faiss-gpu",
]
STRICT_REQUIRED_ADAPTER_COUNT = len(STRICT_REQUIRED_ADAPTERS)


def _write_matrix(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["adapter", "status"])
        writer.writeheader()
        for adapter, status in rows:
            writer.writerow({"adapter": adapter, "status": status})


def _write_portable_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _portable_row(
    *,
    scenario: str,
    budget: str,
    quality: float,
    engine: str = "mock",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    freshness_hit_at_5s: float | None = None,
    evidence_coverage_at_10: float | None = None,
    task_cost_est: float = 1.0,
    errors: int = 0,
    measure_requests: int = 100,
) -> dict:
    payload = {
        "profile": "portable-agentic",
        "budget_level": budget,
        "embedding_model": embedding_model,
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
        "dataset_bundle": "FRAMES_PORTABLE" if scenario == "s3_multi_hop" else "D4",
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


def test_verify_promotion_gate_passes_for_strict_ready_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_portable_promotion_gate_passes_b0_to_b1_and_writes_candidates(tmp_path: Path) -> None:
    results_path = tmp_path / "results.parquet"
    candidates_path = tmp_path / "promotion_candidates.json"
    _write_portable_results(
        results_path,
        [
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
        ],
    )

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
    written = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert written["promoted_survivor_count"] == 3


def test_verify_portable_promotion_gate_rejects_stale_s2_freshness(tmp_path: Path) -> None:
    results_path = tmp_path / "results.parquet"
    _write_portable_results(
        results_path,
        [
            _portable_row(scenario="s1_single_hop", budget="b0", quality=0.19),
            _portable_row(
                scenario="s2_streaming_memory",
                budget="b0",
                quality=0.19,
                freshness_hit_at_5s=0.5,
            ),
            _portable_row(
                scenario="s3_multi_hop",
                budget="b0",
                quality=0.23,
                evidence_coverage_at_10=0.23,
            ),
        ],
    )

    summary = verify_portable_promotion_gate(results_path=results_path, from_budget="b0")

    assert summary["pass"] is False
    assert summary["ready_for_promotion"] is False
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
            _portable_row(
                scenario="s2_streaming_memory",
                budget="b1",
                quality=0.23,
                freshness_hit_at_5s=0.9,
            ),
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
    assert summary["to_budget"] == "b2"
    s1_survivors = [row for row in summary["survivors"] if row["scenario"] == "s1_single_hop"]
    assert len(s1_survivors) == 3
    assert [row["engine"] for row in s1_survivors] == ["engine-0", "engine-1", "engine-2"]


def test_verify_portable_promotion_gate_cli_dispatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_path = tmp_path / "results.parquet"
    _write_portable_results(
        results_path,
        [
            _portable_row(scenario="s1_single_hop", budget="b0", quality=0.19),
            _portable_row(
                scenario="s2_streaming_memory",
                budget="b0",
                quality=0.19,
                freshness_hit_at_5s=0.7,
            ),
            _portable_row(
                scenario="s3_multi_hop",
                budget="b0",
                quality=0.23,
                evidence_coverage_at_10=0.23,
            ),
        ],
    )

    code = cli_main(
        [
            "verify-promotion-gate",
            "--portable-results",
            str(results_path),
            "--from-budget",
            "b0",
            "--json",
        ]
    )

    assert code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["mode"] == "portable-agentic-promotion"
    assert parsed["pass"] is True


def test_verify_promotion_gate_fails_for_nonpassing_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": False,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 1,
        "errors": [{"message": "adapter `qdrant` failed conformance"}],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT - 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert summary["ready_for_promotion"] is False
    assert summary["reasons"]


def test_verify_promotion_gate_passes_for_gpu_omitted_strict_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required),
        "conformance_status_counts": {"pass": len(required)},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_rejects_gpu_omitted_nonpass_counts_without_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required) + 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "provide --conformance-matrix" in " ".join(summary["reasons"])


def test_verify_promotion_gate_passes_gpu_omitted_with_faiss_gpu_nonpass_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required) + 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in required] + [("mock", "pass"), ("faiss-gpu", "fail")]
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []


def test_verify_promotion_gate_rejects_gpu_omitted_nonpass_on_non_gpu_adapter(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    required = [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"]
    payload = {
        "pass": True,
        "required_adapters": required,
        "allow_gpu_unavailable": True,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": len(required) + 2,
        "conformance_status_counts": {"pass": len(required), "fail": 2},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in required if adapter != "qdrant"]
    rows.extend([("qdrant", "fail"), ("mock", "pass"), ("faiss-gpu", "fail")])
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "permits non-pass matrix rows only for `faiss-gpu`" in " ".join(summary["reasons"])


def test_verify_promotion_gate_passes_with_conformance_matrix_cross_check(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_matrix(matrix_path, [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "pass")])
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is True
    assert summary["ready_for_promotion"] is True
    assert summary["reasons"] == []
    assert summary["conformance_matrix_path"] == str(matrix_path.resolve())


def test_verify_promotion_gate_rejects_conformance_matrix_status_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mismatch_rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "pass")]
    mismatch_rows[-1] = (mismatch_rows[-1][0], "fail")
    _write_matrix(matrix_path, mismatch_rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "status_counts disagrees" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_required_adapter_missing_from_matrix(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS]
    # Replace one required adapter with mock to keep row/status counts unchanged.
    rows[-1] = ("mock", "pass")
    _write_matrix(matrix_path, rows)
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "required_adapters missing in conformance matrix" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_mock_pass_row_when_required(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_matrix(matrix_path, [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS])
    summary = verify_promotion_gate(
        strict_readiness_summary_path=summary_path,
        conformance_matrix_path=matrix_path,
    )
    assert summary["pass"] is False
    assert "requires mock pass" in " ".join(summary["reasons"])


def test_verify_promotion_gate_cli_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": False,
        "required_adapters": [],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": False,
        "error_count": 2,
        "errors": [{"message": "missing required adapters"}],
        "conformance_rows": 0,
        "conformance_status_counts": {"fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code = cli_main(["verify-promotion-gate", "--strict-readiness-summary", str(summary_path), "--json"])
    assert code == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pass"] is False
    assert parsed["reasons"]


def test_verify_promotion_gate_cli_missing_matrix_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            str(summary_path),
            "--conformance-matrix",
            str(tmp_path / "missing.csv"),
            "--json",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-promotion-gate failed:" in captured.err


def test_verify_promotion_gate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_promotion_gate(strict_readiness_summary_path=tmp_path / "missing.json")


def test_verify_promotion_gate_missing_matrix_file_raises(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=tmp_path / "missing.csv",
        )


def test_verify_promotion_gate_matrix_missing_required_columns_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path.write_text("adapter,exit_code\nqdrant,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_matrix_empty_status_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("mock", "")]
    _write_matrix(matrix_path, rows)
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_matrix_empty_adapter_raises_value_error(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT + 1,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [(adapter, "pass") for adapter in STRICT_REQUIRED_ADAPTERS] + [("", "pass")]
    _write_matrix(matrix_path, rows)
    with pytest.raises(ValueError):
        verify_promotion_gate(
            strict_readiness_summary_path=summary_path,
            conformance_matrix_path=matrix_path,
        )


def test_verify_promotion_gate_cli_invalid_matrix_columns_returns_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    matrix_path = tmp_path / "conformance_matrix.csv"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path.write_text("adapter,exit_code\nqdrant,0\n", encoding="utf-8")
    code = cli_main(
        [
            "verify-promotion-gate",
            "--strict-readiness-summary",
            str(summary_path),
            "--conformance-matrix",
            str(matrix_path),
            "--json",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verify-promotion-gate failed:" in captured.err


def test_verify_promotion_gate_rejects_allow_nonpass_mode(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": True,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "allow_nonpass_status=true" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_require_mock_pass(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_require_mock_pass_false(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": False,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "require_mock_pass=false" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_allow_gpu_unavailable(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "allow_gpu_unavailable" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_required_adapters_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": [name for name in STRICT_REQUIRED_ADAPTERS if name != "faiss-gpu"],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "required_adapters mismatch" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_non_string_required_adapters(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS[:-1] + [7],
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "required_adapters must contain only strings" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_missing_conformance_status_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "conformance_status_counts" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_nonpass_conformance_status_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT - 1, "fail": 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "includes non-pass rows" in " ".join(summary["reasons"])


def test_verify_promotion_gate_rejects_conformance_status_count_sum_mismatch(tmp_path: Path) -> None:
    summary_path = tmp_path / "engine_readiness_summary.json"
    payload = {
        "pass": True,
        "required_adapters": STRICT_REQUIRED_ADAPTERS,
        "allow_gpu_unavailable": False,
        "allow_nonpass_status": False,
        "require_mock_pass": True,
        "behavior_cards_ok": True,
        "error_count": 0,
        "errors": [],
        "conformance_rows": STRICT_REQUIRED_ADAPTER_COUNT,
        "conformance_status_counts": {"pass": STRICT_REQUIRED_ADAPTER_COUNT + 1},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = verify_promotion_gate(strict_readiness_summary_path=summary_path)
    assert summary["pass"] is False
    assert "row count mismatch" in " ".join(summary["reasons"])
