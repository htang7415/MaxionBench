"""llm-d routing stack as a harness target (no Kubernetes; deploy/llmd-nok8s).

The real llm-d EPP (endpoint picker) and Envoy run in Docker; workers are either native vllm-metal
servers on the Apple GPU or llm-d-inference-sim containers (for scheduler studies at replica counts
the GPU cannot hold). The EPP discovers workers from a rendered endpoints file and scores them from
their vLLM `/metrics` using a selectable scorer profile.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping
import urllib.request

import yaml

from maxionbench.harness.targets import (
    QWEN3_NO_THINKING,
    Target,
    VllmMetal,
    _check_keys,
    port_in_use,
    wait_healthy,
)
from maxionbench.rag.routing import PICKERS, EndpointPicker

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy" / "llmd-nok8s"
DOCKER_HOST_GATEWAY = "192.168.65.254"  # Docker Desktop: containers reach host loopback services here
SIM_IMAGE = "ghcr.io/llm-d/llm-d-inference-sim:v0.11.4"

# Scorer profiles (plugin names from llm-d's optimized-baseline / no-kubernetes guides).
SCORER_PROFILES: dict[str, list[tuple[str, int]]] = {
    "optimized-baseline": [
        ("queue-scorer", 2), ("kv-cache-utilization-scorer", 2), ("prefix-cache-scorer", 3), ("no-hit-lru-scorer", 2),
    ],
    "load-aware": [("queue-scorer", 1), ("kv-cache-utilization-scorer", 1)],
    "prefix-aware": [("prefix-cache-scorer", 1)],
}


def render_epp_config(profile: str) -> dict[str, Any]:
    if profile not in SCORER_PROFILES:
        raise ValueError(f"scorer_profile must be one of {sorted(SCORER_PROFILES)}")
    scorers = SCORER_PROFILES[profile]
    return {
        "apiVersion": "llm-d.ai/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [
            {"name": "file-discovery", "type": "file-discovery",
             "parameters": {"path": "/etc/epp/endpoints.yaml", "watchFile": True}},
            *({"type": name} for name, _ in scorers),
            {"name": "metrics-source", "type": "metrics-data-source"},
            {"name": "metrics-extractor", "type": "core-metrics-extractor"},
        ],
        "schedulingProfiles": [
            {"name": "default", "plugins": [{"pluginRef": name, "weight": weight} for name, weight in scorers]},
        ],
        "dataLayer": {
            "injectDefaults": False,
            "discovery": {"pluginRef": "file-discovery"},
            "sources": [{"pluginRef": "metrics-source", "extractors": [{"pluginRef": "metrics-extractor"}]}],
        },
    }


def render_endpoints(ports: list[int], model: str, address: str = DOCKER_HOST_GATEWAY) -> dict[str, Any]:
    """file-discovery needs literal IPv4 addresses; workers are host-loopback services seen from Docker."""
    return {
        "endpoints": [
            {"name": f"worker-{i}", "address": address, "port": str(port), "labels": {"model": model}}
            for i, port in enumerate(ports)
        ]
    }


class _SimWorkers:
    """N llm-d-inference-sim containers published on 127.0.0.1:<base_port + i>."""

    def __init__(self, replicas: int, base_port: int, model: str, args: list[str]) -> None:
        self.replicas, self.base_port, self.model, self.args = replicas, base_port, model, args
        self.names = [f"maxionbench-sim-{base_port + i}" for i in range(replicas)]
        self.base_urls = [f"http://127.0.0.1:{base_port + i}" for i in range(replicas)]

    def __enter__(self) -> "_SimWorkers":
        try:
            for i, name in enumerate(self.names):
                _docker(["run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{self.base_port + i}:8000",
                         SIM_IMAGE, "--model", self.model, "--port", "8000", *self.args])
            for url in self.base_urls:
                wait_healthy(url, timeout_s=60)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        subprocess.run(["docker", "rm", "-f", *self.names], capture_output=True, check=False)


class LlmdNoK8s(Target):
    kind = "llmd"
    KEYS = {"workers", "worker_params", "scorer_profile", "gateway_port", "model", "disable_thinking"}

    def __init__(self, params: Mapping[str, Any], log_dir: Path) -> None:
        _check_keys(self.kind, params, self.KEYS)
        self.worker_kind = str(params.get("workers", "sim"))
        self.worker_params = dict(params.get("worker_params") or {})
        self.profile = str(params.get("scorer_profile", "optimized-baseline"))
        render_epp_config(self.profile)  # validate early
        self.gateway_port = int(params.get("gateway_port", 8081))
        self.model = str(params.get("model", "qwen3"))
        self.disable_thinking = bool(params.get("disable_thinking", True))
        self.log_dir = log_dir
        self.base_urls = [f"http://127.0.0.1:{self.gateway_port}"]
        if self.worker_kind == "vllm_metal":
            self.workers: Any = VllmMetal(
                {**self.worker_params, "extra_args": [*self.worker_params.get("extra_args", []),
                                                      "--served-model-name", self.model]},
                log_dir / "workers",
            )
        elif self.worker_kind == "sim":
            wp = self.worker_params
            self.workers = _SimWorkers(int(wp.get("replicas", 4)), int(wp.get("base_port", 8300)), self.model,
                                       [str(a) for a in wp.get("args", [])])
        else:
            raise ValueError("llmd.workers must be 'vllm_metal' or 'sim'")
        self.worker_ports = [self.workers.base_port + i for i in range(self.workers.replicas)]
        self._env: dict[str, str] = {}

    def __enter__(self) -> "LlmdNoK8s":
        busy = [p for p in (self.gateway_port, 9090, 19000) if port_in_use(p)]
        if busy:
            raise RuntimeError(f"ports {busy} are already in use; refusing to start llm-d stack")
        run_dir = self.log_dir / "llmd"
        (run_dir / "epp").mkdir(parents=True, exist_ok=True)
        (run_dir / "epp" / "config.yaml").write_text(yaml.safe_dump(render_epp_config(self.profile), sort_keys=False))
        (run_dir / "epp" / "endpoints.yaml").write_text(
            yaml.safe_dump(render_endpoints(self.worker_ports, self.model), sort_keys=False)
        )
        self._env = {**os.environ, "LLMD_RUN_DIR": str(run_dir.resolve()), "GATEWAY_PORT": str(self.gateway_port)}
        try:
            self.workers.__enter__()
            self._compose(["up", "-d"])
            self._wait_gateway(timeout_s=120)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self._env:
            logs = subprocess.run(self._compose_cmd(["logs", "--no-color"]), env=self._env,
                                  capture_output=True, text=True, check=False)
            (self.log_dir / "llmd" / "compose.log").write_text(logs.stdout + logs.stderr)
            subprocess.run(self._compose_cmd(["down", "--remove-orphans"]), env=self._env,
                           capture_output=True, check=False)
        self.workers.__exit__(None, None, None)

    def picker(self) -> EndpointPicker:
        return PICKERS["round_robin"](1)  # routing happens inside llm-d

    def request_options(self) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model}
        if self.disable_thinking:
            body.update(QWEN3_NO_THINKING)
        return {"extra_body": body}

    def describe(self) -> dict[str, Any]:
        compose = yaml.safe_load((DEPLOY_DIR / "compose.yaml").read_text())
        return {
            "kind": self.kind,
            "engine": "llm-d",
            "scorer_profile": self.profile,
            "scorers": SCORER_PROFILES[self.profile],
            "epp_image": compose["services"]["epp"]["image"],
            "envoy_image": compose["services"]["envoy"]["image"],
            "workers": self.worker_kind,
            "worker_ports": self.worker_ports,
            "worker": self.workers.describe() if hasattr(self.workers, "describe") else
            {"image": SIM_IMAGE, "replicas": self.workers.replicas, "args": self.workers.args},
        }

    def _compose_cmd(self, args: list[str]) -> list[str]:
        return ["docker", "compose", "-f", str(DEPLOY_DIR / "compose.yaml"), *args]

    def _compose(self, args: list[str]) -> None:
        out = subprocess.run(self._compose_cmd(args), env=self._env, capture_output=True, text=True, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"docker compose {' '.join(args)} failed: {out.stderr.strip()[-400:]}")

    def _wait_gateway(self, timeout_s: float) -> None:
        """Ready when a tiny request succeeds end to end through Envoy -> EPP -> worker."""
        body = (b'{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
                % self.model.encode())
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            try:
                req = urllib.request.Request(self.base_urls[0] + "/v1/chat/completions", data=body,
                                             headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        return
            except OSError as exc:
                last = str(exc)
            time.sleep(2)
        raise TimeoutError(f"llm-d gateway not serving within {timeout_s}s: {last}")


def _docker(args: list[str]) -> None:
    out = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {out.stderr.strip()[-300:]}")
