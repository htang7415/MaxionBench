"""Workloads: request streams plus the load-generation settings that drive them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from maxionbench.rag.loadgen import RequestSpec

LOADGEN_KEYS = {
    "rate_rps", "concurrency", "max_in_flight", "timeout_s", "max_tokens", "warmup_requests", "ignore_eos",
}


@dataclass(frozen=True)
class Workload:
    specs: list[RequestSpec]
    rate_rps: float | None  # open loop (Poisson arrivals) ...
    concurrency: int | None  # ... or closed loop (fixed number of back-to-back clients)
    max_in_flight: int
    timeout_s: float
    max_tokens: int
    warmup_requests: int = 0  # sent after target start, before measurement; results discarded
    extra_body: dict[str, Any] = field(default_factory=dict)  # workload-level request fields


def warmup_specs(n: int) -> list[RequestSpec]:
    """Short prompts unrelated to any workload, so warm-up cannot seed the prefix cache."""
    return [
        RequestSpec(f"warmup{i}", f"warmup{i}", f"warmup{i}", ({"role": "user", "content": f"Warm-up {i}: say ok."},))
        for i in range(n)
    ]


def make_workload(kind: str, params: Mapping[str, Any], seed: int) -> Workload:
    if kind == "rag_sessions":
        specs = _rag_sessions(params, seed)
    elif kind == "synthetic_chat":
        specs = _synthetic_chat(params)
    else:
        raise ValueError(f"unknown workload kind {kind!r}")
    return Workload(
        specs=specs,
        rate_rps=float(params["rate_rps"]) if "rate_rps" in params else None,
        concurrency=int(params["concurrency"]) if "concurrency" in params else None,
        max_in_flight=int(params.get("max_in_flight", 12)),
        timeout_s=float(params.get("timeout_s", 30.0)),
        max_tokens=int(params.get("max_tokens", 24)),
        warmup_requests=int(params.get("warmup_requests", 0)),
        # ignore_eos: generate exactly max_tokens (fixed output length, as in vLLM's serving benchmark)
        extra_body={"ignore_eos": True} if params.get("ignore_eos") else {},
    )


def _rag_sessions(params: Mapping[str, Any], seed: int) -> list[RequestSpec]:
    """Multi-turn HotpotQA sessions sharing a k-paragraph context (see serving_bench.build_workload)."""
    from maxionbench.tools.serving_bench import build_workload

    _check_keys("rag_sessions", params, LOADGEN_KEYS | {"dataset", "sessions", "turns", "k", "window"})
    return build_workload(
        Path(str(params.get("dataset", "dataset/processed/hotpot_portable"))),
        sessions=int(params.get("sessions", 30)),
        turns=int(params.get("turns", 3)),
        k=int(params.get("k", 5)),
        window=int(params.get("window", 8)),
        seed=seed,
    )


def _synthetic_chat(params: Mapping[str, Any]) -> list[RequestSpec]:
    """Deterministic chat requests; `sessions` controls how many distinct shared prefixes exist."""
    _check_keys("synthetic_chat", params, LOADGEN_KEYS | {"requests", "sessions", "prompt_words"})
    requests = int(params.get("requests", 20))
    sessions = max(1, int(params.get("sessions", 4)))
    words = int(params.get("prompt_words", 200))
    specs = []
    for i in range(requests):
        s = i % sessions
        context = " ".join(f"fact{s}-{w}" for w in range(words))
        specs.append(
            RequestSpec(
                request_id=f"q{i}",
                session_id=f"s{s}",
                prefix_key=f"s{s}",
                messages=(
                    {"role": "system", "content": "Answer briefly."},
                    {"role": "user", "content": f"{context}\n\nQuestion {i}: summarize in one word."},
                ),
                fallback_text="",
            )
        )
    return specs


def _check_keys(kind: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"{kind}: unknown params {sorted(unknown)}")
    if ("rate_rps" in params) == ("concurrency" in params):
        raise ValueError(f"{kind}: set exactly one of rate_rps (open loop) or concurrency (closed loop)")
