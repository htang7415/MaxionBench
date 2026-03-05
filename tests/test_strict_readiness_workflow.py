from __future__ import annotations

from pathlib import Path

import yaml


def test_strict_readiness_workflow_has_dispatch_inputs_and_commands() -> None:
    workflow = Path(".github/workflows/strict_readiness.yml")
    assert workflow.exists()

    text = workflow.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    assert payload.get("name") == "strict-readiness"

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

    jobs = payload.get("jobs", {})
    assert isinstance(jobs, dict)
    assert "strict_readiness_gate" in jobs
    job = jobs["strict_readiness_gate"]
    assert isinstance(job, dict)
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    runs_blob = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))

    assert "maxionbench conformance-matrix" in runs_blob
    assert "--config-dir \"${{ inputs.conformance_config_dir }}\"" in runs_blob
    assert "--out-dir artifacts/conformance_strict" in runs_blob
    assert "--timeout-s \"${{ inputs.timeout_s }}\"" in runs_blob
    assert "maxionbench verify-engine-readiness" in runs_blob
    assert "--conformance-matrix artifacts/conformance_strict/conformance_matrix.csv" in runs_blob
    assert "--behavior-dir docs/behavior" in runs_blob
    assert "--allow-gpu-unavailable" in runs_blob
    assert "--allow-nonpass-status" not in runs_blob
    assert "tee artifacts/conformance_strict/engine_readiness_summary.json" in runs_blob
    assert "strict-readiness-artifacts" in text
    assert "artifacts/conformance_strict/**" in text
    assert "artifacts/conformance_strict/engine_readiness_summary.json" in text
