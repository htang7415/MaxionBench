from __future__ import annotations

import bz2
import json
from pathlib import Path
import zipfile

import pytest

from maxionbench.datasets import sources
from maxionbench.datasets.loaders import v03


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_manifest_pins_every_loader_input() -> None:
    m = sources.manifest()
    pinned = {**m["files"], **m["derived"]}
    needed = {v03.CRAG_SLICE, v03.SHAREGPT_SAMPLE, v03.AZURE_CONV_TRACE, "beir/scifact.zip", "beir/fiqa.zip"}
    needed |= {f"bfcl/{p}BFCL_v3_{c}.json" for c in v03.BFCL_CATEGORIES for p in ("", "possible_answer/")}
    assert needed <= set(pinned)
    for rel, meta in pinned.items():
        assert len(meta["sha256"]) == 64, rel
        assert meta["license"], rel
    for meta in m["derived"].values():
        assert meta["from"] in m["files"] and meta["kind"] in sources.DERIVERS


def test_verified_path_rejects_modified_file(tmp_path: Path) -> None:
    rel = "bfcl/BFCL_v3_simple.json"
    _write_jsonl(tmp_path / rel, [{"id": "tampered"}])
    with pytest.raises(ValueError, match="sha256 mismatch"):
        sources.verified_path(rel, tmp_path)
    assert not (tmp_path / ".sha256-cache.json").exists()  # failures are never cached
    with pytest.raises(FileNotFoundError):
        sources.verified_path(v03.CRAG_SLICE, tmp_path)


def test_crag_slice_drops_html_and_loads(tmp_path: Path) -> None:
    page = {"page_name": "P", "page_url": "u", "page_snippet": "snip", "page_last_modified": "", "page_result": "<html>"}
    rows = [
        {"interaction_id": f"i{n}", "query": f"q{n}", "query_time": "t", "answer": "a", "alt_ans": ["b"],
         "domain": "finance", "question_type": "simple", "static_or_dynamic": "static", "search_results": [page]}
        for n in range(3)
    ]
    src = tmp_path / "crag.jsonl.bz2"
    src.write_bytes(bz2.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    dst = tmp_path / "slice.jsonl"
    sources.derive_crag_slice(src, dst, examples=2)
    examples = v03.load_crag(dst)
    assert [e.interaction_id for e in examples] == ["i0", "i1"]
    assert examples[0].alt_ans == ("b",) and "page_result" not in examples[0].pages[0]


def test_sharegpt_sample_keeps_first_human_turn(tmp_path: Path) -> None:
    data = [
        {"id": "a", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
        {"id": "b", "conversations": [{"from": "human", "value": "only one turn"}]},
        {"id": "c", "conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "y"}]},
    ]
    src = tmp_path / "sharegpt.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    dst = tmp_path / "sample.jsonl"
    sources.derive_sharegpt_sample(src, dst, examples=1)
    assert v03.load_sharegpt(dst) == [v03.ChatPair("a", "hi", "hello")]


def test_beir_reads_from_zip(tmp_path: Path) -> None:
    zpath = tmp_path / "scifact.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("scifact/corpus.jsonl", '{"_id": "d1", "title": "T", "text": "body"}\n')
        zf.writestr("scifact/queries.jsonl", '{"_id": "q1", "text": "query"}\n{"_id": "q2", "text": "unjudged"}\n')
        zf.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    ds = v03.load_beir("scifact", path=zpath)
    assert ds.queries == {"q1": "query"} and ds.qrels == {"q1": {"d1": 1}}
    assert ds.corpus["d1"] == {"title": "T", "text": "body"}


def test_azure_trace_window(tmp_path: Path) -> None:
    lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    lines += [f"2024-05-12 00:00:{s:02d}.500000+00:00,{100 + s},{s}" for s in range(10)]
    path = tmp_path / "trace.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    w = v03.load_azure_trace(2.0, 4.0, path)
    assert w.arrival_s.tolist() == [0.0, 1.0, 2.0, 3.0]  # offsets 2..5 from the first request
    assert w.context_tokens.tolist() == [102, 103, 104, 105] and w.rate_rps == 1.0
    with pytest.raises(ValueError, match="no requests"):
        v03.load_azure_trace(100.0, 1.0, path)


def test_bfcl_joins_questions_and_answers(tmp_path: Path) -> None:
    fn = {"name": "f", "parameters": {"type": "dict", "properties": {}, "required": []}}
    _write_jsonl(tmp_path / "BFCL_v3_simple.json", [{"id": "simple_0", "question": [[{"role": "user", "content": "x"}]],
                                                    "function": [fn]}])
    _write_jsonl(tmp_path / "possible_answer" / "BFCL_v3_simple.json", [{"id": "simple_0", "ground_truth": [{"f": {}}]}])
    (case,) = v03.load_bfcl("simple", root=tmp_path)
    assert case.messages == ({"role": "user", "content": "x"},) and case.ground_truth == ({"f": {}},)


_HAVE_DATA = sources.DATASET_ROOT.exists()


@pytest.mark.skipif(not _HAVE_DATA, reason="dataset/v03 not fetched")
def test_real_datasets_match_manifest() -> None:
    assert len(v03.load_crag()) == 500
    assert len(v03.load_sharegpt()) == 2000
    assert len(v03.load_beir("scifact").queries) == 300
    assert [len(v03.load_bfcl(c)) for c in v03.BFCL_CATEGORIES] == [400, 200, 200, 200]
    window = v03.load_azure_trace(0.0, 60.0)
    assert window.rate_rps > 1 and (window.generated_tokens >= 0).all()
