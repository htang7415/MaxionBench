"""Workloads: request streams plus the load-generation settings that drive them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from maxionbench.rag.loadgen import RequestSpec

LOADGEN_KEYS = {"rate_rps", "max_in_flight", "timeout_s", "max_tokens"}


@dataclass(frozen=True)
class Workload:
    specs: list[RequestSpec]
    rate_rps: float
    max_in_flight: int
    timeout_s: float
    max_tokens: int


def make_workload(kind: str, params: Mapping[str, Any], seed: int) -> Workload:
    if kind == "rag_sessions":
        specs = _rag_sessions(params, seed)
    elif kind == "synthetic_chat":
        specs = _synthetic_chat(params)
    else:
        raise ValueError(f"unknown workload kind {kind!r}")
    return Workload(
        specs=specs,
        rate_rps=float(params["rate_rps"]),
        max_in_flight=int(params.get("max_in_flight", 12)),
        timeout_s=float(params.get("timeout_s", 30.0)),
        max_tokens=int(params.get("max_tokens", 24)),
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
    if "rate_rps" not in params:
        raise ValueError(f"{kind}: rate_rps is required")
