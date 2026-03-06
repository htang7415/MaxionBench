from __future__ import annotations

import numpy as np

from maxionbench.adapters.mock import MockAdapter
from maxionbench.datasets.d3_generator import D3Params
from maxionbench.scenarios.s3_churn_smooth import S3Config, run as run_s3
from maxionbench.schemas.adapter_contract import QueryRequest, QueryResult


class _Top1NoiseAdapter(MockAdapter):
    """Inject one non-relevant top-ranked item to separate recall from rank-aware metrics."""

    def query(self, request: QueryRequest) -> list[QueryResult]:
        rows = super().query(request)
        if not rows:
            return rows
        fake = QueryResult(id="__noise__", score=rows[0].score + 1.0, payload={})
        return [fake, *rows[:-1]]


def test_s3_rank_aware_metrics_are_not_recall_placeholders() -> None:
    adapter = _Top1NoiseAdapter()
    adapter.create(collection="s3-quality-regression", dimension=16, metric="ip")
    cfg = S3Config(
        vector_dim=16,
        num_vectors=600,
        num_queries=30,
        top_k=10,
        sla_threshold_ms=120.0,
        warmup_s=0.0,
        steady_state_s=1.0,
        lambda_req_s=20.0,
        read_rate=20.0,
        insert_rate=0.0,
        update_rate=0.0,
        delete_rate=0.0,
        maintenance_interval_s=60.0,
        phase_timing_mode="strict",
        max_events=40,
    )
    d3_params = D3Params(
        k_clusters=64,
        num_tenants=20,
        num_acl_buckets=8,
        num_time_buckets=16,
        beta_tenant=0.75,
        beta_acl=0.70,
        beta_time=0.65,
        seed=17,
    )

    result = run_s3(
        adapter=adapter,
        cfg=cfg,
        rng=np.random.default_rng(17),
        d3_params=d3_params,
    )

    # With an injected non-relevant rank-1 result, rank-aware metrics should diverge from recall.
    assert result.recall_at_10 > 0.0
    assert result.ndcg_at_10 > 0.0
    assert result.mrr_at_10 > 0.0
    assert result.ndcg_at_10 != result.recall_at_10
    assert result.mrr_at_10 != result.recall_at_10
