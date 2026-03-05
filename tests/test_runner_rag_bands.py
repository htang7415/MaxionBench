from __future__ import annotations

import json

from maxionbench.orchestration.config_schema import RunConfig
from maxionbench.orchestration.runner import _RagCandidate, _select_rag_band_rows


def _candidate(
    *,
    label: str,
    ndcg: float,
    rhu_h: float,
    p99_ms: float,
    qps: float,
) -> _RagCandidate:
    return _RagCandidate(
        label=label,
        search_payload={"mode": label},
        p50_ms=p99_ms / 2.0,
        p95_ms=p99_ms * 0.9,
        p99_ms=p99_ms,
        qps=qps,
        recall_at_10=0.9,
        ndcg_at_10=ndcg,
        mrr_at_10=0.8,
        sla_violation_rate=0.0,
        errors=0,
        rhu_h=rhu_h,
        rtt_baseline_ms_p50=0.1,
        rtt_baseline_ms_p99=0.2,
        setup_elapsed_s=0.01,
        warmup_target_s=120.0,
        warmup_elapsed_s=0.5,
        warmup_requests=10,
        measure_target_s=300.0,
        measure_elapsed_s=1.0,
        measure_requests=10,
        resource_cpu_vcpu=1.0,
        resource_gpu_count=0.0,
        resource_ram_gib=1.0,
        resource_disk_tb=0.01,
        rhu_rate=0.1,
    )


def test_rag_band_boundaries_follow_pinned_ranges() -> None:
    cfg = RunConfig(
        scenario="s5_rerank",
        clients_read=16,
        clients_write=0,
        repeats=1,
        no_retry=True,
        quality_target=0.35,
    )
    rows = _select_rag_band_rows(
        cfg=cfg,
        repeat_idx=0,
        config_fingerprint="cfg",
        candidates=[
            _candidate(label="low-edge", ndcg=0.3499, rhu_h=0.5, p99_ms=10.0, qps=100.0),
            _candidate(label="medium-edge", ndcg=0.35, rhu_h=0.4, p99_ms=9.0, qps=120.0),
            _candidate(label="high-edge", ndcg=0.55, rhu_h=0.3, p99_ms=8.0, qps=140.0),
        ],
        suffix_prefix="s5",
    )
    by_band = {}
    for row in rows:
        payload = json.loads(row.search_params_json)
        by_band[payload["rag_ndcg_band"]] = (row, payload)

    assert set(by_band.keys()) == {"low", "medium", "high"}

    low_row, low_payload = by_band["low"]
    assert low_row.ndcg_at_10 < 0.35
    assert low_payload["rag_ndcg_range"] == [0.0, 0.35]
    assert float(low_row.quality_target) == 0.0

    medium_row, medium_payload = by_band["medium"]
    assert 0.35 <= medium_row.ndcg_at_10 < 0.55
    assert medium_payload["rag_ndcg_range"] == [0.35, 0.55]
    assert float(medium_row.quality_target) == 0.35

    high_row, high_payload = by_band["high"]
    assert high_row.ndcg_at_10 >= 0.55
    assert high_payload["rag_ndcg_range"] == [0.55, 1.0]
    assert float(high_row.quality_target) == 0.55


def test_rag_band_selection_tiebreak_prefers_low_p99_then_high_qps() -> None:
    cfg = RunConfig(
        scenario="s5_rerank",
        clients_read=16,
        clients_write=0,
        repeats=1,
        no_retry=True,
        quality_target=0.35,
    )
    rows = _select_rag_band_rows(
        cfg=cfg,
        repeat_idx=0,
        config_fingerprint="cfg",
        candidates=[
            _candidate(label="m-best", ndcg=0.40, rhu_h=0.2, p99_ms=8.0, qps=120.0),
            _candidate(label="m-same-rhu-worse-p99", ndcg=0.42, rhu_h=0.2, p99_ms=10.0, qps=200.0),
            _candidate(label="m-same-rhu-p99-lower-qps", ndcg=0.44, rhu_h=0.2, p99_ms=8.0, qps=110.0),
        ],
        suffix_prefix="s5",
    )
    assert len(rows) == 1
    payload = json.loads(rows[0].search_params_json)
    assert payload["rag_ndcg_band"] == "medium"
    assert payload["mode"] == "m-best"
