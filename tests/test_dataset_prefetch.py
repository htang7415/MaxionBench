from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import yaml

from maxionbench.orchestration.slurm.dataset_prefetch import prefetch_for_configs


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _make_beir_subset(root: Path, name: str) -> None:
    subset = root / name
    (subset / "qrels").mkdir(parents=True, exist_ok=True)
    (subset / "corpus.jsonl").write_text('{"_id":"d1","text":"doc"}\n', encoding="utf-8")
    (subset / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    (subset / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")


def test_prefetch_for_configs_materializes_d3_from_local_npz(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = _write_yaml(
        repo_root / "configs" / "scenarios_paper" / "calibrate_d3.yaml",
        {
            "scenario": "calibrate_d3",
            "dataset_bundle": "D3",
            "calibration_require_real_data": True,
        },
    )
    source_path = tmp_path / "laion_source.npz"
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    np.savez(source_path, vectors=vectors)

    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=[config_path],
        env_sh_path=repo_root / "artifacts" / "prefetch" / "dataset_env.sh",
        env={"MAXIONBENCH_PREFETCH_D3_SOURCE": str(source_path)},
    )

    target = repo_root / "data" / "d3" / "laion_d3_vectors.npy"
    env_sh = repo_root / "artifacts" / "prefetch" / "dataset_env.sh"
    assert target.exists()
    assert np.load(target).shape == (6, 4)
    assert "d3" in summary["fetched"]
    text = env_sh.read_text(encoding="utf-8")
    assert "MAXIONBENCH_D3_DATASET_PATH" in text
    assert str(target) in text
    assert "MAXIONBENCH_D3_DATASET_SHA256" in text


def test_prefetch_for_configs_reuses_existing_d3_dataset_path_export(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = _write_yaml(
        repo_root / "configs" / "scenarios_paper" / "calibrate_d3.yaml",
        {
            "scenario": "calibrate_d3",
            "dataset_bundle": "D3",
            "calibration_require_real_data": True,
        },
    )
    staged_dataset = repo_root / "shared" / "processed" / "D3" / "yfcc-10M" / "base.npy"
    staged_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.save(staged_dataset, np.arange(24, dtype=np.float32).reshape(6, 4))

    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=[config_path],
        env_sh_path=repo_root / "artifacts" / "prefetch" / "dataset_env.sh",
        env={
            "MAXIONBENCH_D3_DATASET_PATH": str(staged_dataset),
            "MAXIONBENCH_DATASET_ROOT": str(repo_root / "shared"),
        },
    )

    assert summary["fetched"]["d3"]["path"] == str(staged_dataset.resolve())
    assert summary["fetched"]["d3"]["source"] == "env_MAXIONBENCH_D3_DATASET_PATH"
    assert not (repo_root / "data" / "d3" / "laion_d3_vectors.npy").exists()
    text = (repo_root / "artifacts" / "prefetch" / "dataset_env.sh").read_text(encoding="utf-8")
    assert str(staged_dataset.resolve()) in text


def test_prefetch_for_configs_reuses_processed_d3_cache_when_available(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = _write_yaml(
        repo_root / "configs" / "scenarios_paper" / "calibrate_d3.yaml",
        {
            "scenario": "calibrate_d3",
            "dataset_bundle": "D3",
            "calibration_require_real_data": True,
        },
    )
    processed_base = repo_root / "dataset" / "processed" / "D3" / "yfcc-10M" / "base.npy"
    processed_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(processed_base, np.arange(32, dtype=np.float32).reshape(8, 4))

    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=[config_path],
        env_sh_path=repo_root / "artifacts" / "prefetch" / "dataset_env.sh",
    )

    assert summary["fetched"]["d3"]["path"] == str(processed_base.resolve())
    assert summary["fetched"]["d3"]["source"] == "processed_dataset_cache"
    assert not (repo_root / "data" / "d3" / "laion_d3_vectors.npy").exists()


def test_prefetch_for_configs_materializes_d4_from_local_sources(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = _write_yaml(
        repo_root / "configs" / "scenarios_paper" / "s4_hybrid.yaml",
        {
            "scenario": "s4_hybrid",
            "dataset_bundle": "D4",
            "d4_use_real_data": True,
            "d4_beir_root": "data/beir",
            "d4_beir_subsets": ["fiqa", "scifact"],
            "d4_beir_split": "test",
            "d4_crag_path": "data/crag_task_1_and_2_dev_v4.jsonl.bz2",
            "d4_include_crag": True,
        },
    )
    beir_source = tmp_path / "beir_source"
    _make_beir_subset(beir_source, "fiqa")
    _make_beir_subset(beir_source, "scifact")
    crag_source = tmp_path / "crag_source.jsonl.bz2"
    with bz2.open(crag_source, "wt", encoding="utf-8") as handle:
        handle.write('{"query_id":"q1","query":"what","candidates":[{"doc_id":"d1","text":"doc","relevance":1}]}\n')

    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=[config_path],
        env_sh_path=repo_root / "artifacts" / "prefetch" / "dataset_env.sh",
        env={
            "MAXIONBENCH_PREFETCH_D4_BEIR_SOURCE": str(beir_source),
            "MAXIONBENCH_PREFETCH_D4_CRAG_SOURCE": str(crag_source),
        },
    )

    assert (repo_root / "data" / "beir" / "fiqa" / "corpus.jsonl").exists()
    assert (repo_root / "data" / "beir" / "scifact" / "queries.jsonl").exists()
    assert (repo_root / "data" / "crag_task_1_and_2_dev_v4.jsonl.bz2").exists()
    assert "d4_beir" in summary["fetched"]
    assert "d4_crag" in summary["fetched"]
    text = (repo_root / "artifacts" / "prefetch" / "dataset_env.sh").read_text(encoding="utf-8")
    assert "MAXIONBENCH_D4_BEIR_ROOT" in text
    assert "MAXIONBENCH_D4_CRAG_PATH" in text
    assert "MAXIONBENCH_D4_CRAG_SHA256" in text


def test_prefetch_for_configs_reuses_downloaded_d4_repo_cache(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_path = _write_yaml(
        repo_root / "configs" / "scenarios_paper" / "s4_hybrid.yaml",
        {
            "scenario": "s4_hybrid",
            "dataset_bundle": "D4",
            "d4_use_real_data": True,
            "d4_beir_root": "data/beir",
            "d4_beir_subsets": ["fiqa", "scifact"],
            "d4_beir_split": "test",
            "d4_crag_path": "data/crag_task_1_and_2_dev_v4.jsonl.bz2",
            "d4_include_crag": True,
        },
    )
    downloaded_beir_root = repo_root / "dataset" / "D4" / "beir"
    _make_beir_subset(downloaded_beir_root, "fiqa")
    _make_beir_subset(downloaded_beir_root, "scifact")
    downloaded_crag = repo_root / "dataset" / "D4" / "crag" / "crag_task_1_and_2_dev_v4.jsonl.bz2"
    downloaded_crag.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(downloaded_crag, "wt", encoding="utf-8") as handle:
        handle.write('{"query_id":"q1","query":"what","candidates":[{"doc_id":"d1","text":"doc","relevance":1}]}\n')

    summary = prefetch_for_configs(
        repo_root=repo_root,
        config_paths=[config_path],
        env_sh_path=repo_root / "artifacts" / "prefetch" / "dataset_env.sh",
    )

    assert (repo_root / "data" / "beir" / "fiqa" / "corpus.jsonl").exists()
    assert (repo_root / "data" / "beir" / "scifact" / "queries.jsonl").exists()
    assert (repo_root / "data" / "crag_task_1_and_2_dev_v4.jsonl.bz2").exists()
    assert summary["fetched"]["d4_beir"]["source"] == "downloaded_dataset_cache"
    assert summary["fetched"]["d4_crag"]["source"] == "downloaded_dataset_cache"
