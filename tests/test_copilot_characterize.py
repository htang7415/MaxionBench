from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from maxionbench.datasets.loaders import copilot
from maxionbench.datasets.loaders.copilot import iter_sessions
from maxionbench.eval import copilot_characterize as cc

DAY = "2026-06-01"


def call(ts: str, prompt: int | None, cached: int = 0, model: str = "Model A", dur_ms: float = 1000.0,
         segments: tuple[tuple[str, str, int], ...] = ()) -> dict[str, Any]:
    return {
        "timestamp": f"{DAY}T{ts}.000000000Z", "duration_ms": dur_ms, "initiator_type": "agent", "model": model,
        "tokens": {"prompt": prompt, "cached": cached, "completion": 10 if prompt else None},
        "message_metadata": [{"type": t, "role": r, "sequenceId": i, "token_len": n}
                             for i, (t, r, n) in enumerate(segments)],
    }


SESSIONS = [
    {"session_id": "s1", "turns": [
        {"llm_calls": [
            call("10:00:00", 10_000, 0, segments=(("System", "system", 4_000), ("History", "tool", 6_000))),
            call("10:00:02", None),  # tool-model helper: no token accounting
            call("10:00:05", 20_000, 9_000, segments=(  # the batch's two results follow the last assistant message
                ("System", "system", 4_000), ("History", "assistant", 1_000),
                ("FunctionCalls", "tool", 5_000), ("FunctionCalls", "tool", 1_000))),
        ], "tool_batches": [
            {"duration_ms": 500.0, "function_calls": [{"name": "get_file", "status": 1}, {"name": "run_build", "status": 2}]},
        ]},
        {"llm_calls": [
            call("10:10:06", 5_000, 1_000),  # 600 s after the previous call ended; a drop to 25%
            call("10:10:10", 6_000, 4_000, model="Model B"),
        ], "tool_batches": []},
    ]},
    {"session_id": "s2", "turns": [{"llm_calls": [call("11:00:00", 3_000)], "tool_batches": None}]},
]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(copilot, "verified_path", lambda rel, root: Path(root) / rel)
    shard = gzip.compress("".join(json.dumps(s) + "\n" for s in SESSIONS).encode())
    path = tmp_path / copilot.archive(DAY)
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in ((f"date={DAY}/manifest.json", b"{}"), (f"date={DAY}/shard-0000.jsonl.gz", shard)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tmp_path


def test_iter_sessions_streams_shards_and_honours_limit(root: Path) -> None:
    assert [s["session_id"] for s in iter_sessions(DAY, root)] == ["s1", "s2"]
    assert [s["session_id"] for s in iter_sessions(DAY, root, limit=1)] == ["s1"]


def test_summary_measures_context_drops_cache_and_pauses(root: Path) -> None:
    s = cc.summarize(cc.merge([cc.profile_day(DAY, root)]))
    assert {"sessions": 2, "turns": 3, "llm_calls": 5, "calls_without_tokens": 1, "tool_calls": 2}.items() <= s["scale"].items()
    assert s["scale"]["prompt_tokens"] == 44_000 and s["scale"]["cached_tokens"] == 14_000
    assert s["context"]["median_prompt_by_call_index"] == {"1": 6_500.0, "2": 20_000.0, "4": 6_000.0}
    assert s["scale"]["overlapping_pair_share"] == 0.0  # timestamps mark call ends: the chain is sequential
    assert s["composition"]["token_share"] == {"System/system": 0.381, "History/tool": 0.2857,
                                               "FunctionCalls/tool": 0.2857, "History/assistant": 0.0476}
    assert s["composition"]["new_tool_result_tokens"]["count"] == 3  # 6k (first call), then 5k and 1k
    assert s["composition"]["new_tool_result_tokens"]["p50"] == 5_000
    assert s["composition"]["tool_result_token_share_from_results_over"] == {"2k": 0.9167, "8k": 0.0, "32k": 0.0}
    # pairs: 10k->20k (steady, same turn), 20k->5k (drop, new turn), 5k->6k (model switch)
    assert s["drops"]["per_1k_pairs"] == pytest.approx(1000 / 3, abs=1e-3)
    assert s["drops"]["share_sessions"] == 0.5 and s["drops"]["share_within_turn"] == 0.0
    assert s["cache"]["next_cached_frac_mean"] == {"steady": 0.45, "after_drop": 0.2, "after_model_switch": 0.6667,
                                                   "first_call_of_turn": None}  # no steady pair starts a turn
    assert s["pauses_s"]["within_turn"]["p50"] == pytest.approx(3.5)  # 1 s calls: starts 4 s and 3 s after prior ends
    assert s["pauses_s"]["between_turns"]["p50"] == pytest.approx(600.0)
    assert s["tools"]["top"]["run_build"] == {"calls": 1, "failure_rate": 1.0}
