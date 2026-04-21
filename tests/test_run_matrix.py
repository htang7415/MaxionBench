from __future__ import annotations

from pathlib import Path

import yaml

from maxionbench.orchestration.run_matrix import build_run_matrix


def test_build_run_matrix_cpu_lane_filters_gpu_rows_and_generates_configs(tmp_path: Path) -> None:
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_paper"),
        engine_config_dir=Path("configs/engines"),
        out_dir=tmp_path / "cpu_matrix",
        output_root="artifacts/runs/paper_cpu_matrix",
        lane="cpu",
    )

    assert matrix.lane == "cpu"
    assert matrix.cpu_rows
    assert matrix.gpu_rows == []
    assert any(row.engine == "qdrant" for row in matrix.cpu_rows)
    assert all(row.scenario != "s5_rerank" for row in matrix.cpu_rows)

    sample_path = Path(matrix.cpu_rows[0].config_path)
    assert sample_path.exists()
    payload = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert str(payload["output_dir"]).startswith("artifacts/runs/paper_cpu_matrix/")


def test_build_run_matrix_gpu_lane_contains_s5_or_faiss_gpu(tmp_path: Path) -> None:
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_paper"),
        engine_config_dir=Path("configs/engines"),
        out_dir=tmp_path / "gpu_matrix",
        output_root="artifacts/runs/paper_gpu_matrix",
        lane="gpu",
    )

    assert matrix.lane == "gpu"
    assert matrix.cpu_rows == []
    assert matrix.gpu_rows
    assert any(row.scenario == "s5_rerank" or row.engine == "faiss-gpu" for row in matrix.gpu_rows)


def test_build_run_matrix_normalizes_d3_and_d4_dataset_paths(tmp_path: Path) -> None:
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_paper"),
        engine_config_dir=Path("configs/engines"),
        out_dir=tmp_path / "all_matrix",
        output_root="artifacts/runs/paper_all_matrix",
        lane="all",
        skip_s6=True,
    )

    d3_s2 = next(row for row in matrix.cpu_rows if row.scenario == "s2_filtered_ann")
    d3_payload = yaml.safe_load(Path(d3_s2.config_path).read_text(encoding="utf-8"))
    assert d3_payload["dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D3/yfcc-10M/base.npy"

    d4_s4 = next(row for row in matrix.cpu_rows if row.scenario == "s4_hybrid")
    d4_payload = yaml.safe_load(Path(d4_s4.config_path).read_text(encoding="utf-8"))
    assert d4_payload["processed_dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/D4"
    assert all(Path(row.template_name).stem != "s6_fusion" for row in matrix.iter_rows())


def test_build_run_matrix_portable_expands_embedding_variants(tmp_path: Path) -> None:
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines"),
        out_dir=tmp_path / "portable_matrix",
        output_root="artifacts/runs/portable_matrix",
        lane="cpu",
    )

    s1_rows = [row for row in matrix.cpu_rows if row.scenario == "s1_single_hop" and row.engine == "qdrant"]
    assert s1_rows
    payloads = [yaml.safe_load(Path(row.config_path).read_text(encoding="utf-8")) for row in s1_rows]
    models = {str(payload["embedding_model"]) for payload in payloads}
    dims = {int(payload["vector_dim"]) for payload in payloads}
    assert "BAAI/bge-small-en-v1.5" in models
    assert "BAAI/bge-base-en-v1.5" in models
    assert dims == {384, 768}
