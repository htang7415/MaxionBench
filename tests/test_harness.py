from __future__ import annotations

from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import pytest

from maxionbench.harness.compare import compare, load_result
from maxionbench.harness.planner import plan
from maxionbench.harness.results import ExperimentResult, TrialResult, from_dict, mean_ci
from maxionbench.harness.runner import run_experiment
from maxionbench.harness.schema_export import SCHEMA_PATH, result_json_schema, schema_text
from maxionbench.harness.spec import parse_spec
from maxionbench.harness.targets import StaticEndpoints


def _spec(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "maxionbench-experiment-v1",
        "name": "unit",
        "seed": 7,
        "repeats": 2,
        "target": {"kind": "static_endpoints", "params": {"urls": ["http://127.0.0.1:1"]}},
        "workload": {
            "kind": "synthetic_chat",
            "params": {"requests": 4, "sessions": 2, "prompt_words": 5, "rate_rps": 200.0, "max_tokens": 4},
        },
        "slo": {"ttft_s": 1.0, "e2e_s": 2.0},
        "matrix": {"target.routing_policy": ["round_robin", "prefix_affinity"]},
        "quiet_host": {"max_load_1m": 1000.0, "wait_s": 0},
    }
    payload.update(overrides)
    return payload


def test_parse_spec_validates_structure() -> None:
    spec = parse_spec(_spec())
    assert spec.repeats == 2 and spec.seed_strategy == "per_repeat"
    with pytest.raises(ValueError, match="unknown spec keys"):
        parse_spec(_spec(extra=1))
    with pytest.raises(ValueError, match="matrix key"):
        parse_spec(_spec(matrix={"routing_policy": ["round_robin"]}))
    with pytest.raises(ValueError, match="non-empty list"):
        parse_spec(_spec(matrix={"target.routing_policy": []}))
    with pytest.raises(ValueError, match="repeats"):
        parse_spec(_spec(repeats=0))
    with pytest.raises(ValueError, match="schema_version"):
        parse_spec(_spec(schema_version="v0"))


def test_plan_is_repeat_major_shuffled_and_applies_matrix() -> None:
    spec = parse_spec(
        _spec(repeats=3, matrix={"target.routing_policy": ["a", "b"], "workload.rate_rps": [0.5, 1.0]})
    )
    trials = plan(spec)
    assert len(trials) == 12
    for repeat in range(3):
        block = trials[repeat * 4 : repeat * 4 + 4]
        assert {t.repeat for t in block} == {repeat}
        assert len({t.cell_id for t in block}) == 4
        assert {t.seed for t in block} == {7 + repeat}
    orders = [[t.cell_id for t in trials[r * 4 : r * 4 + 4]] for r in range(3)]
    assert len({tuple(o) for o in orders}) > 1  # cell order reshuffled across repeats
    t = next(t for t in trials if t.cell_id == "routing_policy=b,rate_rps=1.0")
    assert t.target_params["routing_policy"] == "b"
    assert t.workload_params["rate_rps"] == 1.0
    assert t.target_params["urls"] == ["http://127.0.0.1:1"]
    assert [x.trial_id for x in plan(spec)] == [x.trial_id for x in trials]
    fixed = plan(parse_spec(_spec(seed_strategy="fixed")))
    assert {x.seed for x in fixed} == {7}


def test_mean_ci_student_t() -> None:
    ci = mean_ci([1.0, 2.0, 3.0])
    assert ci.mean == pytest.approx(2.0)
    assert ci.std == pytest.approx(1.0)
    assert ci.ci_high - ci.mean == pytest.approx(4.303 / 3**0.5, rel=1e-4)
    single = mean_ci([5.0])
    assert (single.ci_low, single.ci_high, single.n) == (5.0, 5.0, 1)


def test_committed_schema_matches_result_model() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == schema_text(), "run: python -m maxionbench.harness schema --write"
    schema = result_json_schema()
    assert schema["required"] == [f.name for f in fields(ExperimentResult)]
    assert schema["$defs"]["TrialResult"]["required"] == [f.name for f in fields(TrialResult)]
    assert schema["$defs"]["TrialResult"]["properties"]["status"] == {"enum": ["ok", "failed"]}


class _FakeLLM(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        time.sleep(0.01)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for event in (
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 5}}},
        ):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def fake_urls() -> Iterator[list[str]]:
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield [f"http://127.0.0.1:{s.server_port}" for s in servers]
    finally:
        for server in servers:
            server.shutdown()


def test_run_experiment_writes_valid_bundle_with_cis(tmp_path: Path, fake_urls: list[str]) -> None:
    spec = parse_spec(_spec(target={"kind": "static_endpoints", "params": {"urls": fake_urls}}))
    out_dir, result = run_experiment(spec, tmp_path, log=lambda msg: None)

    assert {p.name for p in out_dir.iterdir()} >= {"spec.yaml", "results.json", "requests.jsonl"}
    reloaded = load_result(out_dir)
    assert reloaded == from_dict(ExperimentResult, result.to_dict())
    assert reloaded.schema_version == "maxionbench-harness-result-v1"
    assert len(reloaded.trials) == 4 and all(t.status == "ok" for t in reloaded.trials)
    assert reloaded.provenance.git_commit and len(reloaded.provenance.spec_fingerprint) == 64
    for cell in reloaded.cells:
        assert cell.n_ok == 2 and cell.n_failed == 0
        ci = cell.metrics["slo_attainment"]
        assert ci.n == 2 and ci.ci_low <= ci.mean <= ci.ci_high
        assert cell.metrics["prefix_cache_hit_ratio"].mean == pytest.approx(0.25)
    lines = (out_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 * 4
    assert compare(reloaded, reloaded)["overlapping"] == compare(reloaded, reloaded)["compared"] > 0


def test_failed_trial_is_recorded_and_run_continues(tmp_path: Path, fake_urls: list[str]) -> None:
    calls = {"n": 0}

    def flaky_factory(kind: str, params: dict[str, Any], log_dir: Path) -> StaticEndpoints:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("replica failed to start")
        return StaticEndpoints({**params, "urls": fake_urls})

    spec = parse_spec(_spec(repeats=1))
    _, result = run_experiment(spec, tmp_path, target_factory=flaky_factory, log=lambda msg: None)
    statuses = sorted(t.status for t in result.trials)
    assert statuses == ["failed", "ok"]
    failed = next(t for t in result.trials if t.status == "failed")
    assert "replica failed to start" in (failed.error or "")
    assert sum(c.n_failed for c in result.cells) == 1


def test_from_dict_rejects_unknown_and_mistyped_fields(tmp_path: Path, fake_urls: list[str]) -> None:
    spec = parse_spec(_spec(repeats=1, target={"kind": "static_endpoints", "params": {"urls": fake_urls}}))
    _, result = run_experiment(spec, tmp_path, log=lambda msg: None)
    data = result.to_dict()
    with pytest.raises(TypeError, match="missing or unexpected"):
        from_dict(ExperimentResult, {**data, "extra": 1})
    bad = json.loads(json.dumps(data))
    bad["trials"][0]["status"] = "weird"
    with pytest.raises(TypeError, match="not in"):
        from_dict(ExperimentResult, bad)


def test_parse_llama_version_ignores_log_lines() -> None:
    from maxionbench.harness.targets import parse_llama_version

    text = "0.00.000.579 I srv  llama_server: initializing ...\nversion: 0.5.0 (build 11146, commit 7fe450e19)\nbuilt with AppleClang"
    assert parse_llama_version(text) == "0.5.0 (build 11146, commit 7fe450e19)"
    assert parse_llama_version("no version here") == "unknown"


def test_foreign_engine_detection_excludes_own_process_tree() -> None:
    from maxionbench.harness.runner import foreign_engine_pids

    ps = "\n".join(
        [
            "  100     1 python -m maxionbench.harness run spec.yaml",  # the harness itself
            "  101   100 llama-server -m model.gguf --port 8100",  # ours (child)
            "  102   100 /venv/bin/vllm serve model.gguf --port 8200",  # ours (child)
            "  103   102 python -c from multiprocessing.spawn import spawn_main",  # ours (grandchild)
            "  200     1 /venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8200",  # another session
            "  201     1 python -m mlx_lm.server --model x",  # another session
            "  202     1 /opt/homebrew/bin/Python3.12 /h/.venv-vllm-metal/bin/vllm serve m --port 8300",  # another session
            "  300     1 zsh -c pgrep -f 'vllm serve'",  # a shell mentioning the words, not an engine
        ]
    )
    assert foreign_engine_pids(ps, self_pid=100) == [200, 201, 202]


def test_foreign_engine_containers_ignore_own_and_non_engines() -> None:
    from maxionbench.harness.runner import foreign_engine_containers

    ps = "\n".join(
        [
            "memtrace-vllm-cpu\tvllm/vllm-openai-cpu:latest-arm64",
            "maxionbench-sim-8300\tghcr.io/llm-d/llm-d-inference-sim:v0.11.4",
            "maxionbench-llmd-envoy-1\tdocker.io/envoyproxy/envoy:distroless-v1.33.2",
            "postgres\tpgvector/pgvector:0.8.2-pg16-trixie",
            "other-sim\tghcr.io/llm-d/llm-d-inference-sim:v0.9.0",
        ]
    )
    assert foreign_engine_containers(ps) == ["memtrace-vllm-cpu", "other-sim"]


def test_available_memory_parses_vm_stat() -> None:
    from maxionbench.harness.runner import available_memory_gb

    vm = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               65536.\n"
        "Pages active:                            999999.\n"
        "Pages inactive:                           65536.\n"
        "Pages speculative:                            0.\n"
        "Pages purgeable:                              0.\n"
    )
    assert available_memory_gb(vm) == pytest.approx(2.0)
    assert available_memory_gb("garbage") == float("inf")


def test_memory_gate_refuses_to_start_engine(tmp_path: Path, fake_urls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import maxionbench.harness.runner as runner

    monkeypatch.setattr(runner, "available_memory_gb", lambda: 1.5)
    spec = parse_spec(_spec(repeats=1, target={"kind": "static_endpoints", "params": {"urls": fake_urls}},
                            quiet_host={"max_load_1m": 1000, "wait_s": 0, "min_available_gb": 4}))
    _, result = run_experiment(spec, tmp_path, log=lambda msg: None)
    assert all(t.status == "failed" and "insufficient memory" in (t.error or "") for t in result.trials)
