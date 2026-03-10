from __future__ import annotations

from pathlib import Path


def test_readme_mentions_behavior_and_engine_readiness_gates() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "verify-behavior-cards" in text
    assert "verify-engine-readiness" in text
    assert "allow-nonpass-status" in text
    assert "conformance_readiness_gate" in text
    assert "strict_readiness.yml" in text
    assert "scenario_config_dir" in text
    assert "strict_d3_scenario_scale" in text
    assert "require_paper_d3_calibration" in text
    assert "d3_params_path" in text
    assert "configs/scenarios_paper/" in text
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
    assert "device=\"cuda\"" in text
    assert "verify-slurm-plan --json" in text
    assert "verify-slurm-plan --skip-gpu --json" in text
    assert "verify-dataset-manifests" in text
    assert "verify-d3-calibration" in text
    assert "MAXIONBENCH_D3_DATASET_PATH" in text
    assert "MAXIONBENCH_D3_DATASET_SHA256" in text
    assert "reported S2 runs stay benchmark results rather than tuning runs" in text
    assert "Dockerfile" in text
    assert "apptainer build ... docker-daemon://maxionbench:0.1.0" in text
    assert "docker save" in text
    assert "docker-archive://..." in text
    assert "submit-slurm-plan --container-runtime apptainer --container-image <path>" in text
    assert "--container-bind" in text
    assert "--hf-cache-dir" in text
    assert "profiles_local.yaml" in text
    assert "profiles_local.example.yaml" in text
    assert "submit-slurm-plan --dry-run --json" in text
    assert "submit-slurm-plan --prefetch-datasets --dry-run --json" in text
    assert "submit-slurm-plan --skip-gpu --dry-run --json" in text
    assert "submit-slurm-plan --scenario-config-dir configs/scenarios_paper --skip-gpu --dry-run --json" in text
    assert "submit-slurm-plan --scenario-config-dir <dir>" in text
    assert "prefetch_datasets.sh" in text
    assert "MAXIONBENCH_PREFETCH_D3_SOURCE" in text
    assert "MAXIONBENCH_PREFETCH_D4_BEIR_SOURCE" in text
    assert "uses the override file when present and otherwise falls back to its default scenario config" in text
    assert "same override also applies to `calibrate_d3`" in text
    assert "`MAXIONBENCH_CALIBRATE_CONFIG` is explicitly set" in text
    assert "D3-50M runs reuse the frozen D3 calibration affinities from the 10M paper calibration" in text
    assert "robustness-accounting support baseline, not a headline S1 D1/D2 result" in text
    assert "GPU-omitted mode (`--skip-gpu` / `allow_gpu_unavailable`) omits the GPU array entirely" in text
    assert "validate-slurm-snapshots --json" in text
    assert "slurm_snapshot_validation.json" in text
    assert "ci-protocol-audit --strict" in text
    assert "ci_protocol_audit.json" in text
