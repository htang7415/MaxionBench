"""Systems under test. A target owns its lifecycle, exposes endpoint URLs and a picker, and
describes itself for provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping
import urllib.request

from maxionbench.rag.routing import PICKERS, EndpointPicker

_SHA_CACHE: dict[tuple[str, int, float], str] = {}


class Target:
    base_urls: list[str]

    def __enter__(self) -> "Target":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def picker(self) -> EndpointPicker:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


class StaticEndpoints(Target):
    """Already-running OpenAI-compatible endpoints (e.g. a gateway or a compose stack)."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        _check_keys("static_endpoints", params, {"urls", "routing_policy"})
        self.base_urls = [str(u).rstrip("/") for u in params["urls"]]
        if not self.base_urls:
            raise ValueError("static_endpoints.urls must be non-empty")
        self.routing_policy = str(params.get("routing_policy", "round_robin"))
        _check_policy(self.routing_policy)

    def picker(self) -> EndpointPicker:
        return PICKERS[self.routing_policy](len(self.base_urls))

    def describe(self) -> dict[str, Any]:
        return {"kind": "static_endpoints", "urls": self.base_urls, "routing_policy": self.routing_policy}


class LlamaCppReplicas(Target):
    """N CPU-only llama.cpp replicas started fresh per trial; routing is client-side."""

    KEYS = {
        "model", "replicas", "threads", "slots", "ctx", "cache_ram_mib", "base_port",
        "routing_policy", "llama_server",
    }

    def __init__(self, params: Mapping[str, Any], log_dir: Path) -> None:
        _check_keys("llamacpp_replicas", params, self.KEYS)
        self.model = Path(str(params["model"])).expanduser()
        self.replicas = int(params.get("replicas", 3))
        self.threads = int(params.get("threads", 3))
        self.slots = int(params.get("slots", 2))
        self.ctx = int(params.get("ctx", 8192))
        self.cache_ram_mib = int(params.get("cache_ram_mib", 64))
        self.base_port = int(params.get("base_port", 8100))
        self.routing_policy = str(params.get("routing_policy", "round_robin"))
        self.llama_server = str(params.get("llama_server", "llama-server"))
        _check_policy(self.routing_policy)
        if not self.model.is_file():
            raise FileNotFoundError(f"model not found: {self.model}")
        self.log_dir = log_dir
        self.base_urls = [f"http://127.0.0.1:{self.base_port + i}" for i in range(self.replicas)]
        self.procs: list[subprocess.Popen[bytes]] = []

    def command(self, i: int) -> list[str]:
        return [
            self.llama_server, "-m", str(self.model),
            "--device", "none", "-ngl", "0", "--no-op-offload",  # CPU only, no Metal offload
            "-t", str(self.threads), "-np", str(self.slots), "-c", str(self.ctx),
            "--cache-ram", str(self.cache_ram_mib), "--metrics",
            "--host", "127.0.0.1", "--port", str(self.base_port + i),
        ]

    def __enter__(self) -> "LlamaCppReplicas":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(self.replicas):
                log = (self.log_dir / f"replica{i}.log").open("ab")
                self.procs.append(subprocess.Popen(self.command(i), stdout=log, stderr=subprocess.STDOUT))
            for url in self.base_urls:
                wait_healthy(url, timeout_s=120)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs = []

    def picker(self) -> EndpointPicker:
        return PICKERS[self.routing_policy](self.replicas)

    def describe(self) -> dict[str, Any]:
        version = subprocess.run([self.llama_server, "--version"], capture_output=True, text=True, check=False)
        return {
            "kind": "llamacpp_replicas",
            "model": self.model.name,
            "model_sha256": file_sha256(self.model),
            "llama_server_version": parse_llama_version(version.stdout + "\n" + version.stderr),
            "replicas": self.replicas,
            "threads": self.threads,
            "slots": self.slots,
            "ctx": self.ctx,
            "cache_ram_mib": self.cache_ram_mib,
            "routing_policy": self.routing_policy,
            "command": self.command(0)[1:],
        }


def parse_llama_version(text: str) -> str:
    """Extract e.g. '0.5.0 (build 11146, commit 7fe450e19)' from `llama-server --version` output."""
    for line in text.splitlines():
        if line.strip().startswith("version:"):
            return line.split("version:", 1)[1].strip()
    return "unknown"


def make_target(kind: str, params: Mapping[str, Any], log_dir: Path) -> Target:
    if kind == "llamacpp_replicas":
        return LlamaCppReplicas(params, log_dir)
    if kind == "static_endpoints":
        return StaticEndpoints(params)
    raise ValueError(f"unknown target kind {kind!r}")


def wait_healthy(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"endpoint {url} did not become healthy within {timeout_s}s")


def file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime)
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def _check_keys(kind: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"{kind}: unknown params {sorted(unknown)}")


def _check_policy(policy: str) -> None:
    if policy not in PICKERS:
        raise ValueError(f"routing_policy must be one of {sorted(PICKERS)}")
