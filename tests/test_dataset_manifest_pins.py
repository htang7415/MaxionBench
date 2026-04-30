from __future__ import annotations

from pathlib import Path

import yaml


def test_d4_manifest_pins_crag_source_file_and_url() -> None:
    payload = yaml.safe_load(Path("maxionbench/datasets/manifests/d4.yaml").read_text(encoding="utf-8"))
    assert payload["crag_source"] == "facebookresearch/CRAG"
    assert payload["crag_file"] == "data/crag_task_1_and_2_dev_v4.jsonl.bz2"
    assert payload["crag_url"] == (
        "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
    )
    assert payload["beir_subsets"] == ["scifact", "fiqa"]
    assert payload["crag_slice_queries"] == 500
    assert int(payload["crag_slice_docs"]) > 0
