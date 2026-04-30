from __future__ import annotations

from pathlib import Path

from maxionbench.orchestration.config_schema import load_run_config
import yaml

from maxionbench.orchestration.run_matrix import build_run_matrix


def test_build_run_matrix_portable_expands_embedding_variants_and_saves_budget(tmp_path: Path) -> None:
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
        out_dir=tmp_path / "portable_matrix",
        output_root="artifacts/runs/portable_matrix",
        budget_level="b0",
        lane="cpu",
    )

    assert matrix.lane == "cpu"
    assert matrix.budget_level == "b0"
    assert matrix.gpu_rows == []
    assert matrix.selected_engines == ["faiss-cpu", "lancedb-inproc", "lancedb-service", "pgvector", "qdrant"]
    assert len(matrix.cpu_rows) == 30

    s1_rows = [row for row in matrix.cpu_rows if row.scenario == "s1_single_hop" and row.engine == "qdrant"]
    assert s1_rows
    payloads = [yaml.safe_load(Path(row.config_path).read_text(encoding="utf-8")) for row in s1_rows]
    models = {str(payload["embedding_model"]) for payload in payloads}
    dims = {int(payload["vector_dim"]) for payload in payloads}
    assert models == {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"}
    assert dims == {384, 768}
    assert all(payload["clients_grid"] == [1, 4, 8] for payload in payloads)
    assert all(payload["budget_level"] == "b0" for payload in payloads)


def test_build_run_matrix_default_paths_are_portable(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(Path(".").resolve())
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
        out_dir=tmp_path / "matrix",
        lane="cpu",
    )

    assert {row.scenario for row in matrix.cpu_rows} == {"s1_single_hop", "s2_streaming_memory", "s3_multi_hop"}
    s3_row = next(row for row in matrix.cpu_rows if row.scenario == "s3_multi_hop")
    s3_payload = yaml.safe_load(Path(s3_row.config_path).read_text(encoding="utf-8"))
    assert s3_payload["processed_dataset_path"] == "${MAXIONBENCH_DATASET_ROOT:-dataset}/processed/hotpot_portable"


def test_build_run_matrix_lancedb_inproc_config_expands_env_uri(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MAXIONBENCH_LANCEDB_INPROC_URI", str((tmp_path / "scratch" / "lancedb").resolve()))
    matrix = build_run_matrix(
        repo_root=Path("."),
        scenario_config_dir=Path("configs/scenarios_portable"),
        engine_config_dir=Path("configs/engines_portable"),
        out_dir=tmp_path / "portable_matrix",
        lane="cpu",
    )

    row = next(item for item in matrix.cpu_rows if item.engine == "lancedb-inproc")
    cfg = load_run_config(Path(row.config_path))

    assert cfg.adapter_options["uri"] == str((tmp_path / "scratch" / "lancedb").resolve())
