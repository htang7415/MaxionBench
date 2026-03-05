from __future__ import annotations

from pathlib import Path

import yaml


def test_publish_benchmark_bundle_workflow_enforces_strict_readiness_gate() -> None:
    workflow = Path(".github/workflows/publish_benchmark_bundle.yml")
    assert workflow.exists()

    text = workflow.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert payload.get("name") == "publish-benchmark-bundle"

    on_block = payload.get("on", payload.get(True, {}))
    assert isinstance(on_block, dict)
    assert "workflow_dispatch" in on_block
    dispatch = on_block["workflow_dispatch"]
    assert isinstance(dispatch, dict)
    inputs = dispatch.get("inputs", {})
    assert isinstance(inputs, dict)
    assert "conformance_config_dir" in inputs
    assert "timeout_s" in inputs
    assert "allow_gpu_unavailable" in inputs
    assert "results_dir" in inputs
    assert "figures_dir" in inputs
    assert "bundle_name" in inputs

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "strict_readiness_proof" in jobs
    assert "publish_result_bundle" in jobs

    strict_job = jobs["strict_readiness_proof"]
    assert isinstance(strict_job, dict)
    strict_steps = strict_job.get("steps", [])
    assert isinstance(strict_steps, list)
    strict_blob = "\n".join(str(step.get("run", "")) for step in strict_steps if isinstance(step, dict))
    assert "maxionbench conformance-matrix" in strict_blob
    assert "--out-dir artifacts/conformance_publish" in strict_blob
    assert "maxionbench verify-engine-readiness" in strict_blob
    assert "--conformance-matrix artifacts/conformance_publish/conformance_matrix.csv" in strict_blob
    assert "--allow-nonpass-status" not in strict_blob
    assert "engine_readiness_summary.json" in text
    assert "strict-readiness-proof" in text

    publish_job = jobs["publish_result_bundle"]
    assert isinstance(publish_job, dict)
    assert publish_job.get("needs") == "strict_readiness_proof"
    publish_steps = publish_job.get("steps", [])
    assert isinstance(publish_steps, list)
    publish_blob = "\n".join(str(step.get("run", "")) for step in publish_steps if isinstance(step, dict))
    assert "maxionbench verify-promotion-gate" in publish_blob
    assert "--strict-readiness-summary artifacts/promotion/engine_readiness_summary.json" in publish_blob
    assert "tar -czf artifacts/promotion/benchmark_result_bundle.tgz" in publish_blob
    assert "actions/download-artifact@v4" in text
