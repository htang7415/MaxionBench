from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from typing import Any

import pytest

from maxionbench.agents.context import MASK_TEXT
from maxionbench.harness.gateway import AIGateway
from maxionbench.kvsim.gateway_replay import Renderer, replay, session_calls
from maxionbench.kvsim.live import Arrival
from tests.test_kvsim_copilot import call


def copilot_session(results: int = 5, pause_end: str = "10:10:00") -> dict[str, Any]:
    base = [(0, "System", "system", 300), (1, "UserMessage", "user", 40)]
    segs, calls = list(base), []
    for e in range(results):
        segs = segs + [(2 + 2 * e, "FunctionCalls", "assistant", 16), (3 + 2 * e, "FunctionCalls", "tool", 800)]
        calls.append(call(f"10:00:{10 * (e + 1):02d}", sum(s[3] for s in segs), list(segs)))
    calls.append(call(pause_end, sum(s[3] for s in segs), list(segs) + [(99, "UserMessage", "user", 40)]))
    return {"session_id": "s1", "turns": [{"llm_calls": calls, "tool_batches": []}]}


def test_session_calls_keep_order_cap_idle_gaps_and_first_token_counts() -> None:
    raw = copilot_session()
    raw["turns"][0]["llm_calls"][2]["message_metadata"][0]["token_len"] = 280  # drifted count for the same message
    s = session_calls(raw, idle_cap_s=60.0)
    assert [round(c.t, 3) for c in s.requests[:3]] == [0.0, 10.0, 20.0]
    assert s.requests[-1].t == pytest.approx(42.0 + 60.0)  # the 9-minute idle gap after the call ending at 42 s, capped to 60 s
    assert all(c.messages[0] == ("system", "0|System|system", 300) for c in s.requests)
    assert [len(c.messages) for c in s.requests] == [4, 6, 8, 10, 12, 13]


def test_renderer_is_deterministic_scaled_and_keyed_by_message() -> None:
    r = Renderer(scale=0.25)
    s = session_calls(copilot_session(), idle_cap_s=300.0)
    a, b = r.messages(s.id, s.requests[0]), Renderer(0.25).messages(s.id, s.requests[1])
    assert a == b[:len(a)]  # an unchanged message is the same text in every call (prefix reuse on the wire)
    assert len(a[0]["content"].split()) == 75 and len(a[3]["content"].split()) == 200
    assert a[3]["role"] == "tool" and a[3]["tool_call_id"]
    assert r.text("other", "0|System|system", 300) != a[0]["content"]


class _Upstream(BaseHTTPRequestHandler):
    seen: list[list[dict[str, Any]]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append(body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        usage = {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 40}}
        for chunk in ({"choices": [{"delta": {"content": "x"}}]}, {"choices": [], "usage": usage}):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain not installed")
@pytest.mark.parametrize("arm", [{"policy": "off"}, {"policy": "mask+cache", "keep": 2, "budget_tokens": 600},
                                 {"policy": "mask+cache", "keep": 2, "budget_tokens": 100_000, "pause_s": 0.2}])
def test_replay_through_the_gateway_context_manager(arm: dict[str, Any], tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Upstream.seen = []
    session = session_calls(copilot_session(pause_end="10:01:10"), idle_cap_s=300.0)
    # compress the trace: calls 50 ms apart, the last one after a 0.5 s pause
    arrivals = [Arrival(t=0.05 * i + (0.5 if i == len(session.requests) - 1 else 0), session=0, request=c)
                for i, c in enumerate(session.requests)]
    try:
        gw = AIGateway({"local": {"kind": "static_endpoints", "params": {"urls": [f"http://127.0.0.1:{server.server_port}"]}},
                        "policy": "local_only", "port": 18091, "context": arm}, tmp_path / "gw")
        with gw:
            outcomes = replay(arrivals, [session], gw.base_urls[0], Renderer(0.25), output_scale=1.0,
                              max_output_tokens=4, timeout_s=10)
    finally:
        server.shutdown()
    assert all(o.status == "ok" for o in outcomes) and outcomes[0].cached_tokens == 40
    actions = [o.action for o in sorted(outcomes, key=lambda o: o.scheduled_s)]
    masked = [sum(m["content"] == MASK_TEXT for m in msgs) for msgs in _Upstream.seen]
    if arm["policy"] == "off":
        assert actions == ["off"] * 6 and max(masked) == 0
    elif "pause_s" in arm:
        assert actions == ["start"] + ["append"] * 4 + ["pause_edit"] and masked[-1] == 3
    else:
        assert actions[0] == "start" and "edit" in actions and max(masked) > 0
