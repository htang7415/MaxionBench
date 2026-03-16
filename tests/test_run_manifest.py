from __future__ import annotations

import json
from pathlib import Path

import yaml

from maxionbench.orchestration.slurm.run_manifest import build_run_manifest, resolve_manifest_row


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_build_run_manifest_writes_engine_scenario_matrix(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    engine_dir = tmp_path / "engines"
    out_dir = tmp_path / "out"

    _write_yaml(
        scenario_dir / "s1_ann_frontier_d3.yaml",
        {
            "engine": "mock",
            "engine_version": "0.1.0",
            "scenario": "s1_ann_frontier",
            "dataset_bundle": "D3",
            "dataset_path": "${MAXIONBENCH_D3_DATASET_PATH}",
            "output_dir": "artifacts/runs/paper/s1_ann_frontier_d3_mock",
        },
    )
    _write_yaml(
        scenario_dir / "s3_churn_smooth.yaml",
        {
            "engine": "mock",
            "engine_version": "0.1.0",
            "scenario": "s3_churn_smooth",
            "dataset_bundle": "D3",
            "dataset_path": "${MAXIONBENCH_D3_DATASET_PATH}",
            "output_dir": "artifacts/runs/paper/s3_churn_smooth_mock",
        },
    )
    _write_yaml(
        scenario_dir / "s4_hybrid.yaml",
        {
            "engine": "mock",
            "engine_version": "0.1.0",
            "scenario": "s4_hybrid",
            "dataset_bundle": "D4",
            "output_dir": "artifacts/runs/paper/s4_hybrid_mock",
        },
    )
    _write_yaml(
        scenario_dir / "s5_rerank.yaml",
        {
            "engine": "mock",
            "engine_version": "0.1.0",
            "scenario": "s5_rerank",
            "dataset_bundle": "D4",
            "output_dir": "artifacts/runs/paper/s5_rerank_mock",
        },
    )

    _write_yaml(engine_dir / "qdrant.yaml", {"engine": "qdrant", "adapter_options": {"host": "127.0.0.1", "port": 6333}})
    _write_yaml(engine_dir / "faiss_gpu.yaml", {"engine": "faiss-gpu", "adapter_options": {"gpu_count": 1}})

    manifest = build_run_manifest(
        repo_root=tmp_path,
        scenario_config_dir=scenario_dir,
        engine_config_dir=engine_dir,
        out_dir=out_dir,
        include_gpu=True,
        skip_s6=False,
    )

    assert len(manifest.cpu_rows) == 3
    assert len(manifest.gpu_rows) == 5
    assert manifest.selected_engines == ["faiss-gpu", "qdrant"] or manifest.selected_engines == ["qdrant", "faiss-gpu"]

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["cpu_rows"]) == len(manifest.cpu_rows)
    assert len(payload["gpu_rows"]) == len(manifest.gpu_rows)

    s1_qdrant = next(row for row in manifest.cpu_rows if row.engine == "qdrant" and row.template_name == "s1_ann_frontier_d3.yaml")
    s1_qdrant_cfg = yaml.safe_load(Path(s1_qdrant.config_path).read_text(encoding="utf-8"))
    assert s1_qdrant_cfg["processed_dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D3/yfcc-10M"
    assert "dataset_path" not in s1_qdrant_cfg

    s3_qdrant = next(row for row in manifest.cpu_rows if row.engine == "qdrant" and row.template_name == "s3_churn_smooth.yaml")
    s3_qdrant_cfg = yaml.safe_load(Path(s3_qdrant.config_path).read_text(encoding="utf-8"))
    assert s3_qdrant_cfg["processed_dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D3/yfcc-10M"

    s4_qdrant = next(row for row in manifest.cpu_rows if row.engine == "qdrant" and row.template_name == "s4_hybrid.yaml")
    s4_qdrant_cfg = yaml.safe_load(Path(s4_qdrant.config_path).read_text(encoding="utf-8"))
    assert s4_qdrant_cfg["processed_dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D4"

    resolved = resolve_manifest_row(manifest_path=manifest_path, group="gpu", task_id=0)
    assert resolved.group == "gpu"
    assert resolved.template_name == "s1_ann_frontier_d3.yaml"
