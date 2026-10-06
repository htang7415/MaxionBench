from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Iterator

import pytest

from maxionbench.eval.tool_client import chat_tools
from maxionbench.harness.dashboard_export import export, latest_results


class _Fake(BaseHTTPRequestHandler):
    received: list[dict[str, Any]] = []
    status = 200

    def do_POST(self) -> None:  # noqa: N802
        type(self).received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b'{"error": "nope"}')
            return
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'},
             "extra_content": {"google": {"thought_signature": "sig"}}}]}
        body = {"choices": [{"message": message}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 97,
                          "prompt_tokens_details": {"cached_tokens": 8}}}
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def fake() -> Iterator[str]:
    _Fake.received, _Fake.status = [], 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def test_chat_tools_keeps_message_verbatim_and_bills_reasoning(fake: str) -> None:
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    r = chat_tools(fake, [{"role": "user", "content": "q"}], max_tokens=16, timeout_s=5, tools=tools,
                   extra_body={"model": "m"})
    assert r.status == "ok" and r.ttft_s is None
    message = json.loads(r.text)
    assert message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert (r.prompt_tokens, r.cached_tokens, r.completion_tokens, r.reasoning_tokens) == (40, 8, 7, 50)
    sent = _Fake.received[0]
    assert sent["tools"] == tools and sent["model"] == "m" and sent["max_tokens"] == 16 and "stream" not in sent


def test_chat_tools_reports_http_errors_without_raising(fake: str) -> None:
    _Fake.status = 429
    r = chat_tools(fake, [{"role": "user", "content": "q"}], max_tokens=4, timeout_s=5)
    assert r.status == "error" and r.error.startswith("http 429")
    r = chat_tools("http://127.0.0.1:1", [], max_tokens=4, timeout_s=2)
    assert r.status == "error" and r.prompt_tokens == 0


def _bundle(root: Path, run_id: str, name: str, done: int, planned: int, valid: bool = True) -> None:
    out = root / run_id
    out.mkdir(parents=True)
    result = {
        "schema_version": "maxionbench-harness-result-v1", "run_id": run_id, "name": name, "description": "",
        "spec": {}, "trials": [], "cells": [],
        "provenance": {"git_commit": "abc", "git_dirty": False, "spec_fingerprint": "f", "started_at": "s",
                       "finished_at": "2026-10-06T00:00:00Z", "host": {},
                       "tools": {"trials_planned": planned, "trials_completed": done}},
    }
    if not valid:
        del result["description"]
    (out / "results.json").write_text(json.dumps(result), encoding="utf-8")


def test_export_picks_latest_complete_run_and_validates(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _bundle(runs, "20261001T000000Z-e2", "e2-prefix-caching", 12, 12)
    _bundle(runs, "20261002T000000Z-e2", "e2-prefix-caching", 12, 12)
    _bundle(runs, "20261003T000000Z-e2", "e2-prefix-caching", 5, 12)  # still running: skipped
    _bundle(runs, "20261002T000000Z-x", "not-published", 1, 1)  # not a dashboard experiment
    assert {k: v.parent.name for k, v in latest_results((runs,)).items()} == {
        "e2-prefix-caching": "20261002T000000Z-e2"}
    index = export(tmp_path / "data", (runs,))
    assert [(e["name"], e["page"], e["run_id"]) for e in index["experiments"]] == [
        ("e2-prefix-caching", "caching", "20261002T000000Z-e2")]
    assert (tmp_path / "data" / "e2-prefix-caching.json").exists()
    _bundle(runs, "20261004T000000Z-e2", "e2-prefix-caching", 1, 1, valid=False)
    with pytest.raises(TypeError, match="missing or unexpected"):
        export(tmp_path / "data2", (runs,))
