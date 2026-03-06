from __future__ import annotations

from pathlib import Path


def test_readme_mentions_protocol_payload_requirements() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "validate --input artifacts/runs --strict-schema --enforce-protocol --json" in text
    assert "--enforce-protocol` also validates per-row robustness payloads:" in text
    assert "S2 requires `search_params_json` keys `selectivity`, `filter`, `p99_inflation_vs_unfiltered`" in text
    assert "explicit 100% anchor row (`selectivity=1.0`) with inflation `1.0`" in text
    assert "S3/S3b require `search_params_json` keys `s1_baseline_p99_ms`" in text
    assert "S5 requires `search_params_json.reranker.backend=\"hf_cross_encoder\"`" in text
    assert "`uses_qrels_supervision=false`" in text
    assert "`runtime_errors=0`" in text
    assert "`p99_inflation_vs_s1_baseline`" in text
    assert "`burst_clock_anchor`" in text
    assert "`burst_clock_anchor=\"measurement_start\"`" in text
    assert "`rtt_baseline_request_profile=\"healthcheck_plus_query_topk1_zero_vector\"`" in text
    assert "`dataset_cache_checksums` provenance entries" in text
    assert "T3_robustness_summary.csv" in text
    assert "`p99_inflation_valid_rows`" in text
    assert "`p99_inflation_nan_rows`" in text
    assert "`p99_inflation_status` in `{computed_all_rows, computed_partial_rows, not_computable}`" in text
