"""Go AI gateway (gateway/) as a harness target.

The gateway fronts a local fleet (any other managed target: vllm-metal replicas, an llm-d stack over
sim or Metal workers) and overflows to Gemini under a hard spend cap. The gateway process alone holds
the API key and enforces the cap against the shared ledger, so this target declares no pricing of its
own and the Python side never touches the key.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from maxionbench.harness.budget import BUDGET_DIR_ENV
from maxionbench.harness.targets import Target, _check_keys, make_target, port_in_use, wait_healthy
from maxionbench.rag.routing import PICKERS, EndpointPicker

REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_DIR = REPO_ROOT / "gateway"
DEFAULT_BINARY = GATEWAY_DIR / "bin" / "maxion-gateway"


def ensure_binary(path: Path = DEFAULT_BINARY) -> Path:
    """Build the gateway if the binary is missing or older than any Go source file."""
    sources = list(GATEWAY_DIR.rglob("*.go")) + [GATEWAY_DIR / "go.mod", GATEWAY_DIR / "go.sum"]
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    if not path.exists() or path.stat().st_mtime < newest:
        out = subprocess.run(["go", "build", "-o", str(path), "./cmd/maxion-gateway"], cwd=GATEWAY_DIR,
                             capture_output=True, text=True, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"go build failed: {out.stderr.strip()[-400:]}")
    return path


def scrape_gateway_metrics(base_url: str) -> dict[str, Any]:
    """Route decisions by backend/reason and committed remote spend from the gateway's /metrics."""
    import urllib.request

    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8", "replace")
    except OSError:
        return {}
    routes: dict[str, float] = {}
    spend = 0.0
    for line in text.splitlines():
        if line.startswith("maxion_gateway_route_decisions_total{"):
            labels, value = line.rsplit(" ", 1)
            parts = dict(kv.split("=", 1) for kv in labels[labels.index("{") + 1 : -1].split(","))
            routes[f"{parts['backend'].strip(chr(34))}:{parts['reason'].strip(chr(34))}"] = float(value)
        elif line.startswith("maxion_gateway_remote_spend_usd_total "):
            spend = float(line.rsplit(" ", 1)[1])
    return {"route_decisions": routes, "remote_spend_usd": round(spend, 6)}


class AIGateway(Target):
    kind = "ai_gateway"
    KEYS = {"local", "policy", "failover", "max_inflight", "port", "remote", "local_model", "binary"}

    def __init__(self, params: Mapping[str, Any], log_dir: Path) -> None:
        _check_keys(self.kind, params, self.KEYS)
        local = dict(params.get("local") or {})
        if "kind" not in local:
            raise ValueError("ai_gateway.local must be a target spec with kind and params")
        self.inner = make_target(str(local["kind"]), dict(local.get("params") or {}), log_dir / "local")
        self.policy = str(params.get("policy", "local_first"))
        self.failover = bool(params.get("failover", True))
        self.max_inflight = int(params.get("max_inflight", 8))
        self.port = int(params.get("port", 8090))
        self.remote = dict(params.get("remote") or {})
        self.local_model = params.get("local_model")
        self.binary = Path(str(params.get("binary", DEFAULT_BINARY))).expanduser()
        self.log_dir = log_dir
        self.base_urls = [f"http://127.0.0.1:{self.port}"]
        self.proc: subprocess.Popen[bytes] | None = None

    def render_config(self) -> dict[str, Any]:
        ledger_dir = Path(os.environ.get(BUDGET_DIR_ENV) or Path.home() / ".maxionbench" / "budget")
        remote_enabled = bool(self.remote.get("enabled", False))
        cfg: dict[str, Any] = {
            "listen": f"127.0.0.1:{self.port}",
            "policy": self.policy,
            "failover": self.failover,
            "local": {"upstreams": self.inner.base_urls, "max_inflight": self.max_inflight, "timeout_s": 300},
            "budget": {"ledger_path": str(ledger_dir / "gemini_ledger.jsonl")},  # same ledger as Python
            "remote": {"enabled": remote_enabled},
        }
        if self.local_model:
            cfg["local"]["model"] = str(self.local_model)
        if remote_enabled:
            cfg["remote"].update({
                "base_url": str(self.remote.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")),
                "chat_path": "/chat/completions",
                "model": str(self.remote["model"]),
                "reasoning_effort": str(self.remote.get("reasoning_effort", "minimal")),
                "pricing_file": str(REPO_ROOT / "configs" / "pricing" / "gemini.yaml"),
                "key_file": str(REPO_ROOT / "docs" / "gemini_api.txt"),
                "timeout_s": float(self.remote.get("timeout_s", 120)),
            })
        return cfg

    def __enter__(self) -> "AIGateway":
        if port_in_use(self.port):
            raise RuntimeError(f"port {self.port} is already in use; refusing to start the gateway")
        binary = ensure_binary(self.binary) if self.binary == DEFAULT_BINARY else self.binary
        self.log_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.log_dir / "gateway.yaml"
        cfg_path.write_text(yaml.safe_dump(self.render_config(), sort_keys=False))
        self.inner.__enter__()
        try:
            log = (self.log_dir / "gateway.log").open("ab")
            self.proc = subprocess.Popen([str(binary), "-config", str(cfg_path)], stdout=log, stderr=subprocess.STDOUT)
            wait_healthy(self.base_urls[0], timeout_s=30, procs=[self.proc])
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.inner.__exit__(None, None, None)

    def picker(self) -> EndpointPicker:
        return PICKERS["round_robin"](1)

    def request_options(self) -> dict[str, Any]:
        return self.inner.request_options()  # engine fields pass through; the gateway strips them for Gemini

    def collect(self) -> dict[str, Any]:
        return {"gateway": scrape_gateway_metrics(self.base_urls[0]), "local": self.inner.collect()}

    def describe(self) -> dict[str, Any]:
        cfg = self.render_config()
        cfg["remote"].pop("key_file", None)  # the path is harmless, but keep provenance about behaviour only
        return {"kind": self.kind, "engine": "maxion-gateway", "config": cfg, "local": self.inner.describe()}
