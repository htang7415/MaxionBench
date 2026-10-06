from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Iterator

import pytest

from maxionbench.harness.budget import BudgetLedger
from maxionbench.harness.runner import run_experiment
from maxionbench.harness.spec import parse_spec

FAKE_KEY = "FAKE-gateway-key-abcdefghijklmnop"


def test_python_ledger_reads_go_written_events(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    # Lines exactly as gateway/internal/budget writes them (RFC3339 "Z", release without label).
    path.write_text(
        '{"at":"2026-10-06T00:00:00Z","event":"reserve","reservation_id":"g1","label":"gateway/local_saturated","estimate_usd":0.5}\n'
        '{"at":"2026-10-06T00:00:01Z","event":"commit","reservation_id":"g1","label":"gateway/local_saturated","actual_usd":0.1,"usage":{"input_tokens":10}}\n'
        '{"at":"2026-10-06T00:00:02Z","event":"reserve","reservation_id":"g2","label":"gateway/policy","estimate_usd":0.2}\n'
        '{"at":"2026-10-06T00:00:03Z","event":"release","reservation_id":"g2"}\n',
        encoding="utf-8",
    )
    ledger = BudgetLedger(1.0, path)
    assert ledger.spent_usd() == pytest.approx(0.1)
    assert ledger.remaining_usd() == pytest.approx(0.9)


def _sse(handler: BaseHTTPRequestHandler, text: str, usage: dict[str, Any] | None) -> None:
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.end_headers()
    handler.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode())
    if usage:
        handler.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    handler.wfile.write(b"data: [DONE]\n\n")


class _SlowLocal(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        time.sleep(0.4)
        _sse(self, "local", {"prompt_tokens": 50, "completion_tokens": 1})

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _FakeGemini(BaseHTTPRequestHandler):
    auth: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        type(self).auth.append(self.headers.get("authorization", ""))
        self.rfile.read(int(self.headers["content-length"]))
        _sse(self, "remote", {"prompt_tokens": 1000, "completion_tokens": 10})

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def servers() -> Iterator[tuple[str, str]]:
    local = ThreadingHTTPServer(("127.0.0.1", 0), _SlowLocal)
    remote = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGemini)
    for s in (local, remote):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{local.server_port}", f"http://127.0.0.1:{remote.server_port}"
    finally:
        local.shutdown()
        remote.shutdown()


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain not installed")
def test_gateway_overflow_end_to_end_with_shared_ledger(
    servers: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_url, remote_url = servers
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("MAXIONBENCH_BUDGET_DIR", str(tmp_path / "budget"))
    _FakeGemini.auth = []
    spec = parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "gateway-e2e",
            "repeats": 1,
            "target": {
                "kind": "ai_gateway",
                "params": {
                    "local": {"kind": "static_endpoints", "params": {"urls": [local_url]}},
                    "policy": "local_first",
                    "max_inflight": 1,
                    "port": 18090,
                    "remote": {"enabled": True, "model": "gemini-3.5-flash-lite", "base_url": remote_url},
                },
            },
            "workload": {"kind": "synthetic_chat", "params": {"requests": 9, "concurrency": 3, "max_tokens": 8}},
            "slo": {"ttft_s": 5, "e2e_s": 5},
            "quiet_host": {"max_load_1m": 1000, "wait_s": 0},
        }
    )
    out_dir, result = run_experiment(spec, tmp_path / "runs", log=lambda msg: None)
    trial = result.trials[0]
    assert trial.status == "ok", trial.error
    assert trial.metrics["ok"] == 9
    routes = trial.target["collected"]["gateway"]["route_decisions"]
    assert routes.get("local:capacity", 0) >= 1 and routes.get("remote:local_saturated", 0) >= 1
    assert sum(routes.values()) == 9
    assert all(a == f"Bearer {FAKE_KEY}" for a in _FakeGemini.auth) and _FakeGemini.auth
    # The Go gateway billed the ledger; the Python ledger sees the same spend.
    spent = BudgetLedger(10.0).spent_usd()
    per_request = (1000 * 0.30 + 10 * 2.50) / 1e6
    assert spent == pytest.approx(routes["remote:local_saturated"] * per_request)
    assert trial.target["collected"]["gateway"]["remote_spend_usd"] == pytest.approx(spent, abs=1e-6)
    for path in out_dir.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(encoding="utf-8", errors="replace"), path
