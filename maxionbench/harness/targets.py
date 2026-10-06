"""Systems under test. A target owns its lifecycle, exposes endpoint URLs and a picker, describes
itself for provenance, and declares per-request options and pricing (paid targets only)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Mapping
import urllib.request

from maxionbench.harness.budget import ModelPrice, load_prices
from maxionbench.harness.secrets import Secret, load_gemini_key
from maxionbench.rag.routing import PICKERS, EndpointPicker

_SHA_CACHE: dict[tuple[str, int, float], str] = {}
QWEN3_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


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

    def request_options(self) -> dict[str, Any]:
        """Keyword arguments for `chat_completion` (extra_body, headers, chat_path)."""
        return {}

    def pricing(self) -> tuple[str, ModelPrice, float] | None:
        """(model, price, budget cap) for paid targets; None for local ones."""
        return None

    def collect(self) -> dict[str, Any]:
        """Server-side observations gathered after a trial, before teardown (default: none)."""
        return {}


VLLM_COUNTERS = (
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def scrape_vllm_counters(base_url: str, timeout_s: float = 5.0) -> dict[str, float]:
    """Sum selected vLLM Prometheus counters across label sets; {} if unreachable."""
    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", "replace")
    except OSError:
        return {}
    totals = {name: 0.0 for name in VLLM_COUNTERS}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in totals:
            try:
                totals[name] += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return {k.split(":", 1)[1]: v for k, v in totals.items()}


class StaticEndpoints(Target):
    """Already-running OpenAI-compatible endpoints (e.g. a gateway or a compose stack)."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        _check_keys("static_endpoints", params, {"urls", "routing_policy", "extra_body"})
        self.base_urls = [str(u).rstrip("/") for u in params["urls"]]
        if not self.base_urls:
            raise ValueError("static_endpoints.urls must be non-empty")
        self.routing_policy = str(params.get("routing_policy", "round_robin"))
        self.extra_body = dict(params.get("extra_body") or {})
        _check_policy(self.routing_policy)

    def picker(self) -> EndpointPicker:
        return PICKERS[self.routing_policy](len(self.base_urls))

    def describe(self) -> dict[str, Any]:
        return {"kind": "static_endpoints", "urls": self.base_urls, "routing_policy": self.routing_policy}

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": self.extra_body} if self.extra_body else {}


class ManagedServers(Target):
    """N local server processes started fresh per trial; routing is client-side."""

    kind = "managed"
    startup_timeout_s = 120.0

    def __init__(self, replicas: int, base_port: int, routing_policy: str, log_dir: Path) -> None:
        _check_policy(routing_policy)
        self.replicas = replicas
        self.base_port = base_port
        self.routing_policy = routing_policy
        self.log_dir = log_dir
        self.base_urls = [f"http://127.0.0.1:{base_port + i}" for i in range(replicas)]
        self.procs: list[subprocess.Popen[bytes]] = []

    def command(self, i: int) -> list[str]:
        raise NotImplementedError

    def __enter__(self) -> "ManagedServers":
        # A port already in use means a foreign server would answer our health checks and we would
        # silently benchmark the wrong process; refuse to start instead.
        busy = [self.base_port + i for i in range(self.replicas) if port_in_use(self.base_port + i)]
        if busy:
            raise RuntimeError(f"ports {busy} are already in use by another process; refusing to start {self.kind}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(self.replicas):
                log = (self.log_dir / f"replica{i}.log").open("ab")
                self.procs.append(subprocess.Popen(self.command(i), stdout=log, stderr=subprocess.STDOUT))
            for url in self.base_urls:
                wait_healthy(url, timeout_s=self.startup_timeout_s, procs=self.procs)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs = []

    def picker(self) -> EndpointPicker:
        return PICKERS[self.routing_policy](self.replicas)


class LlamaCppReplicas(ManagedServers):
    """llama.cpp `llama-server` replicas on CPU only (`device: cpu`) or the Metal GPU (`device: metal`)."""

    kind = "llamacpp_replicas"
    KEYS = {
        "model", "replicas", "threads", "slots", "ctx", "cache_ram_mib", "base_port",
        "routing_policy", "llama_server", "device", "disable_thinking", "cache_prompt",
    }

    def __init__(self, params: Mapping[str, Any], log_dir: Path) -> None:
        _check_keys(self.kind, params, self.KEYS)
        super().__init__(
            int(params.get("replicas", 3)), int(params.get("base_port", 8100)),
            str(params.get("routing_policy", "round_robin")), log_dir,
        )
        self.model = Path(str(params["model"])).expanduser()
        self.threads = int(params.get("threads", 3))
        self.slots = int(params.get("slots", 2))
        self.ctx = int(params.get("ctx", 8192))
        self.cache_ram_mib = int(params.get("cache_ram_mib", 64))
        self.llama_server = str(params.get("llama_server", "llama-server"))
        self.device = str(params.get("device", "cpu"))
        self.disable_thinking = bool(params.get("disable_thinking", False))
        self.cache_prompt = bool(params.get("cache_prompt", True))
        if self.device not in ("cpu", "metal"):
            raise ValueError("llamacpp_replicas.device must be 'cpu' or 'metal'")
        if not self.model.is_file():
            raise FileNotFoundError(f"model not found: {self.model}")

    def command(self, i: int) -> list[str]:
        if self.device == "cpu":
            placement = ["--device", "none", "-ngl", "0", "--no-op-offload"]  # no Metal offload at all
        else:
            placement = ["-ngl", "999"]  # all layers on the Metal GPU
        return [
            self.llama_server, "-m", str(self.model), *placement,
            "-t", str(self.threads), "-np", str(self.slots), "-c", str(self.ctx),
            "--cache-ram", str(self.cache_ram_mib), "--metrics", "--jinja",
            "--host", "127.0.0.1", "--port", str(self.base_port + i),
        ]

    def request_options(self) -> dict[str, Any]:
        body: dict[str, Any] = dict(QWEN3_NO_THINKING) if self.disable_thinking else {}
        if not self.cache_prompt:
            body["cache_prompt"] = False
        return {"extra_body": body} if body else {}

    def describe(self) -> dict[str, Any]:
        version = subprocess.run([self.llama_server, "--version"], capture_output=True, text=True, check=False)
        return {
            "kind": self.kind,
            "engine": "llama.cpp",
            "device": self.device,
            "model": self.model.name,
            "model_sha256": file_sha256(self.model),
            "llama_server_version": parse_llama_version(version.stdout + "\n" + version.stderr),
            "replicas": self.replicas,
            "threads": self.threads,
            "slots": self.slots,
            "ctx": self.ctx,
            "cache_ram_mib": self.cache_ram_mib,
            "cache_prompt": self.cache_prompt,
            "routing_policy": self.routing_policy,
            "command": self.command(0)[1:],
        }


class VllmMetal(ManagedServers):
    """vLLM with the vllm-metal plugin (MLX/Metal GPU) on Apple Silicon."""

    kind = "vllm_metal"
    startup_timeout_s = 600.0
    KEYS = {
        "model", "tokenizer", "replicas", "base_port", "routing_policy", "vllm", "max_model_len",
        "gpu_memory_utilization", "enable_prefix_caching", "max_num_seqs", "disable_thinking", "extra_args",
    }

    def __init__(self, params: Mapping[str, Any], log_dir: Path) -> None:
        _check_keys(self.kind, params, self.KEYS)
        super().__init__(
            int(params.get("replicas", 1)), int(params.get("base_port", 8200)),
            str(params.get("routing_policy", "round_robin")), log_dir,
        )
        model = str(params["model"])
        self.model_path = Path(model).expanduser()
        self.model = str(self.model_path) if self.model_path.is_file() else model  # local GGUF or HF id
        self.tokenizer = params.get("tokenizer")
        self.vllm = str(Path(str(params.get("vllm", "~/.venv-vllm-metal/bin/vllm"))).expanduser())
        self.max_model_len = int(params.get("max_model_len", 8192))
        self.gpu_memory_utilization = float(params.get("gpu_memory_utilization", 0.5))
        self.enable_prefix_caching = bool(params.get("enable_prefix_caching", True))
        self.max_num_seqs = int(params.get("max_num_seqs", 16))
        self.disable_thinking = bool(params.get("disable_thinking", True))
        self.extra_args = [str(a) for a in params.get("extra_args") or []]

    def command(self, i: int) -> list[str]:
        cmd = [
            self.vllm, "serve", self.model,
            "--host", "127.0.0.1", "--port", str(self.base_port + i),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--enable-prefix-caching" if self.enable_prefix_caching else "--no-enable-prefix-caching",
            "--enable-prompt-tokens-details",  # report cached prompt tokens in usage
        ]
        if self.tokenizer:
            cmd += ["--tokenizer", str(self.tokenizer)]
        return cmd + self.extra_args

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": dict(QWEN3_NO_THINKING)} if self.disable_thinking else {}

    def describe(self) -> dict[str, Any]:
        version = subprocess.run(
            [str(Path(self.vllm).with_name("python")), "-c",
             "import vllm, vllm_metal; print('MAXIONBENCH_VERSIONS', vllm.__version__, "
             "getattr(vllm_metal, '__version__', '?'))"],
            capture_output=True, text=True, check=False,
        )
        # vLLM may log to stdout on import, so read only the marked line.
        marked = [ln for ln in version.stdout.splitlines() if ln.startswith("MAXIONBENCH_VERSIONS")]
        parts = marked[-1].split()[1:] if marked else []
        return {
            "kind": self.kind,
            "engine": "vllm-metal",
            "device": "metal",
            "model": self.model_path.name if self.model_path.is_file() else self.model,
            "model_sha256": file_sha256(self.model_path) if self.model_path.is_file() else None,
            "vllm_version": parts[0] if parts else "unknown",
            "vllm_metal_version": parts[1] if len(parts) > 1 else "unknown",
            "replicas": self.replicas,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_num_seqs": self.max_num_seqs,
            "routing_policy": self.routing_policy,
            "command": self.command(0)[1:],
        }


class GeminiTarget(Target):
    """Gemini through its OpenAI-compatible endpoint; spend is capped by the budget ledger."""

    kind = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    KEYS = {"model", "reasoning_effort", "pricing_path"}

    def __init__(self, params: Mapping[str, Any]) -> None:
        _check_keys(self.kind, params, self.KEYS)
        self.model = str(params["model"])
        self.reasoning_effort = params.get("reasoning_effort")
        table = load_prices(Path(str(params.get("pricing_path", "configs/pricing/gemini.yaml"))))
        self.price = table.price(self.model)  # refuse unpriced models before any request
        self.cap_usd = table.budget_cap_usd
        self.price_source = f"{table.source} ({table.retrieved})"
        self.base_urls = [self.BASE_URL]

    def picker(self) -> EndpointPicker:
        return PICKERS["round_robin"](1)

    def request_options(self) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return {
            "extra_body": body,
            # The whole header value is a Secret, so it is revealed only inside chat_completion.
            "headers": {"authorization": Secret(f"Bearer {load_gemini_key().reveal()}")},
            "chat_path": "/chat/completions",
        }

    def pricing(self) -> tuple[str, ModelPrice, float]:
        return self.model, self.price, self.cap_usd

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "engine": "gemini-api",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "price_per_m": {"input": self.price.input_per_m, "output": self.price.output_per_m,
                            "cached_input": self.price.cached_input_per_m},
            "price_source": self.price_source,
        }


def make_target(kind: str, params: Mapping[str, Any], log_dir: Path) -> Target:
    if kind == LlamaCppReplicas.kind:
        return LlamaCppReplicas(params, log_dir)
    if kind == VllmMetal.kind:
        return VllmMetal(params, log_dir)
    if kind == GeminiTarget.kind:
        return GeminiTarget(params)
    if kind == "static_endpoints":
        return StaticEndpoints(params)
    if kind == "llmd":
        from maxionbench.harness.llmd import LlmdNoK8s  # local import: llmd builds on this module

        return LlmdNoK8s(params, log_dir)
    raise ValueError(f"unknown target kind {kind!r}")


def wait_healthy(url: str, timeout_s: float, procs: list[subprocess.Popen[bytes]] | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for proc in procs or []:
            if proc.poll() is not None:
                raise RuntimeError(f"server process exited with code {proc.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"endpoint {url} did not become healthy within {timeout_s}s")


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def parse_llama_version(text: str) -> str:
    """Extract e.g. '0.5.0 (build 11146, commit 7fe450e19)' from `llama-server --version` output."""
    for line in text.splitlines():
        if line.strip().startswith("version:"):
            return line.split("version:", 1)[1].strip()
    return "unknown"


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
