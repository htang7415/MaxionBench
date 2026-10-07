from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
import subprocess
import sys
import threading
from typing import Any

import pytest

from maxionbench.harness.budget import BudgetExceededError, BudgetLedger, cost_usd, load_prices
from maxionbench.harness.runner import run_experiment
from maxionbench.harness.secrets import (
    REDACTED,
    MissingSecretError,
    Secret,
    gemini_key_present,
    load_gemini_key,
    redact,
)
from maxionbench.harness.spec import parse_spec
from maxionbench.harness.targets import StaticEndpoints

FAKE_KEY = "FAKE-test-key-0123456789abcdefghij"


def test_key_loading_prefers_env_then_file(tmp_path: Path) -> None:
    key_file = tmp_path / "k.txt"
    key_file.write_text(f"  {FAKE_KEY}\n", encoding="utf-8")
    assert load_gemini_key({"GEMINI_API_KEY": "env-key"}).reveal() == "env-key"
    assert load_gemini_key({"MAXIONBENCH_GEMINI_KEY_FILE": str(key_file)}).reveal() == FAKE_KEY
    missing = {"MAXIONBENCH_GEMINI_KEY_FILE": str(tmp_path / "nope.txt")}
    with pytest.raises(MissingSecretError):
        load_gemini_key(missing)
    assert gemini_key_present(missing) is False


def test_secret_never_renders_or_serializes() -> None:
    secret = Secret(FAKE_KEY)
    for rendered in (repr(secret), str(secret), f"{secret}", f"{secret!r}", "%s" % secret):
        assert FAKE_KEY not in rendered
    with pytest.raises(TypeError):
        json.dumps({"k": secret})
    with pytest.raises(TypeError):
        pickle.dumps(secret)
    assert redact(f"Bearer {FAKE_KEY} failed", secret) == f"Bearer {REDACTED} failed"


class _LeakyServer(BaseHTTPRequestHandler):
    """Simulates a provider error that echoes the credential back."""

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        body = f"invalid key {FAKE_KEY}".encode()
        self.send_response(401)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_key_never_reaches_result_bundle_or_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LeakyServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    calls = {"n": 0}

    def factory(kind: str, params: dict[str, Any], log_dir: Path) -> StaticEndpoints:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(f"auth failed for {FAKE_KEY}")
        return StaticEndpoints({"urls": [url]})

    spec = parse_spec(
        {
            "schema_version": "maxionbench-experiment-v1",
            "name": "leak",
            "repeats": 2,
            "target": {"kind": "static_endpoints", "params": {"urls": [url]}},
            "workload": {"kind": "synthetic_chat", "params": {"requests": 3, "rate_rps": 100.0}},
            "slo": {"ttft_s": 1, "e2e_s": 1},
            "quiet_host": {"max_load_1m": 1000, "wait_s": 0},
        }
    )
    logs: list[str] = []
    try:
        out_dir, result = run_experiment(spec, tmp_path, target_factory=factory, log=logs.append)
    finally:
        server.shutdown()

    assert result.provenance.tools["gemini_key_present"] is True
    assert any(t.status == "failed" for t in result.trials)
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    assert {p.name for p in files} >= {"results.json", "requests.jsonl", "spec.yaml"}
    for path in files:
        assert FAKE_KEY not in path.read_text(encoding="utf-8", errors="replace"), path
    assert REDACTED in (out_dir / "requests.jsonl").read_text(encoding="utf-8")
    assert all(FAKE_KEY not in line for line in logs)


def test_cost_model_bills_cached_tokens_at_cached_rate() -> None:
    table = load_prices()
    price = table.price("gemini-3.5-flash-lite")
    assert table.budget_cap_usd == 20.0
    assert cost_usd(price, input_tokens=1_000_000, output_tokens=0) == pytest.approx(0.30)
    assert cost_usd(price, input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=500_000) == pytest.approx(
        0.5 * 0.30 + 0.5 * 0.03 + 2.50
    )
    assert cost_usd(price, input_tokens=1_000_000, output_tokens=1_000_000, batch=True) == pytest.approx(1.40)
    with pytest.raises(ValueError):
        cost_usd(price, input_tokens=10, output_tokens=0, cached_tokens=11)
    with pytest.raises(KeyError):
        table.price("unpriced-model")


def test_ledger_enforces_cap_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(10.0, path)
    a = ledger.reserve(6.0, "run-a")
    with pytest.raises(BudgetExceededError):
        ledger.reserve(4.5, "run-b")  # open reservation counts against the cap
    ledger.commit(a, 2.0, {"input_tokens": 100})
    b = ledger.reserve(7.5, "run-b")  # 2.0 spent -> 8.0 remaining
    ledger.release(b)
    assert ledger.spent_usd() == pytest.approx(2.0)
    assert ledger.remaining_usd() == pytest.approx(8.0)
    reopened = BudgetLedger(10.0, path)  # spend survives a new process/instance
    assert reopened.spent_usd() == pytest.approx(2.0)
    with pytest.raises(BudgetExceededError):
        reopened.reserve(8.01, "too-big")


def test_ledger_cap_holds_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    script = (
        "import sys; from pathlib import Path\n"
        "from maxionbench.harness.budget import BudgetLedger, BudgetExceededError\n"
        "l = BudgetLedger(10.0, Path(sys.argv[1]))\n"
        "ok = 0\n"
        "for _ in range(10):\n"
        "    try:\n"
        "        l.reserve(1.0, 'p'); ok += 1\n"
        "    except BudgetExceededError:\n"
        "        pass\n"
        "print(ok)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True) for _ in range(4)]
    granted = sum(int(p.communicate(timeout=60)[0].strip()) for p in procs)
    assert granted == 10  # 4 processes x 10 attempts, but only $10 of $1 reservations may exist
    assert BudgetLedger(10.0, path).remaining_usd() == pytest.approx(0.0)
