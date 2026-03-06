from __future__ import annotations

from pathlib import Path


def test_readme_mentions_behavior_and_engine_readiness_gates() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "verify-behavior-cards" in text
    assert "verify-engine-readiness" in text
    assert "allow-nonpass-status" in text
    assert "conformance_readiness_gate" in text
    assert "strict_readiness.yml" in text
    assert ".[dev,engines]" in text
    assert "non-pass rows fail readiness except `faiss-gpu` when `--allow-gpu-unavailable` is active" in text
    assert "verify-promotion-gate" in text
    assert "cross-checks both strict summary and downloaded conformance matrix" in text
    assert "requires a `mock` row with `status=pass` in the matrix artifact" in text
    assert "non-pass rows are allowed only for `faiss-gpu`" in text
    assert "publish_benchmark_bundle.yml" in text
    assert "include-strict-readiness-check" in text
    assert "include-publish-bundle-check" in text
    assert "pre-run-gate" in text
    assert "pre-run-gate --config ci_s1_smoke.yaml --json" in text
    assert "verify-conformance-configs --config-dir configs/conformance --json" in text
    assert "s5_require_hf_backend: true" in text
    assert "MAXIONBENCH_ENABLE_HF_RERANKER" in text
    assert "visible NVIDIA GPU" in text
    assert "verify-slurm-plan --json" in text
    assert "verify-slurm-plan --skip-gpu --json" in text
    assert "verify-dataset-manifests" in text
    assert "verify-d3-calibration" in text
    assert "submit-slurm-plan --dry-run --json" in text
    assert "submit-slurm-plan --skip-gpu --dry-run --json" in text
    assert "validate-slurm-snapshots --json" in text
    assert "slurm_snapshot_validation.json" in text
    assert "ci-protocol-audit --strict" in text
    assert "ci_protocol_audit.json" in text
