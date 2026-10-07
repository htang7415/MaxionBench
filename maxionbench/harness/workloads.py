"""Workloads: request streams plus the load-generation settings that drive them."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
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
    arrivals: list[float] | None = None  # explicit open-loop schedule (trace replay), seconds from start


def warmup_specs(n: int) -> list[RequestSpec]:
    """Short prompts unrelated to any workload, so warm-up cannot seed the prefix cache."""
    return [
        RequestSpec(f"warmup{i}", f"warmup{i}", f"warmup{i}", ({"role": "user", "content": f"Warm-up {i}: say ok."},))
        for i in range(n)
    ]


def make_workload(kind: str, params: Mapping[str, Any], seed: int) -> Workload:
    if kind == "trace_replay":
        return _trace_replay(params, seed)
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


FOLLOW_UPS = (
    "Which document numbers support your answer? Reply with the numbers only.",
    "Quote the single most relevant sentence from the documents.",
    "Name one entity from the documents that is relevant to the question.",
)


def build_workload(dataset_dir: Path, *, sessions: int, turns: int, k: int, window: int, seed: int) -> list[RequestSpec]:
    """Sessions share a k-paragraph context (gold evidence + random distractors) across turns."""
    from maxionbench.eval.qa import build_messages

    rng = random.Random(seed)
    docs: dict[str, str] = {}
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs[row["doc_id"]] = row["text"]
    gold: dict[str, list[str]] = {}
    with (dataset_dir / "qrels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, doc_id, _ = line.rstrip("\n").split("\t")
            gold.setdefault(qid, []).append(doc_id)
    questions = []
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["query_id"] in gold:
                questions.append(row)
    doc_ids = list(docs)
    chosen = rng.sample(questions, sessions)
    per_session: list[list[RequestSpec]] = []
    for s, q in enumerate(chosen):
        ctx_ids = gold[q["query_id"]][:k]
        ctx_ids += [d for d in rng.sample(doc_ids, k) if d not in ctx_ids][: k - len(ctx_ids)]
        rng.shuffle(ctx_ids)
        ctx = [docs[d] for d in ctx_ids]
        prompts = [q["text"], *FOLLOW_UPS][:turns]
        per_session.append(
            [
                RequestSpec(
                    request_id=f"s{s}-t{t}",
                    session_id=f"s{s}",
                    prefix_key=f"s{s}",
                    messages=tuple(build_messages(prompt, ctx)),
                    fallback_text=ctx[0][:200],  # retrieval-only degraded answer
                )
                for t, prompt in enumerate(prompts)
            ]
        )
    # Interleave: `window` sessions are active at once; each request is the next turn of a random active session.
    order: list[RequestSpec] = []
    pending = list(range(sessions))
    active = [pending.pop(0) for _ in range(min(window, sessions))]
    cursor = [0] * sessions
    while active:
        s = rng.choice(active)
        order.append(per_session[s][cursor[s]])
        cursor[s] += 1
        if cursor[s] == len(per_session[s]):
            active.remove(s)
            if pending:
                active.append(pending.pop(0))
    return order


def _rag_sessions(params: Mapping[str, Any], seed: int) -> list[RequestSpec]:
    """Multi-turn HotpotQA sessions sharing a k-paragraph context (see build_workload)."""
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


TRACE_KEYS = {
    "start_s", "duration_s", "rate_scale", "token_scale", "max_prompt_tokens", "max_output_tokens",
    "max_in_flight", "timeout_s", "warmup_requests", "ignore_eos",
}
# Short common words, roughly one BPE token each; servers report the actual prompt_tokens per request.
_FILLER = (
    "the of and to in is was for on as with by at from his her that it an be this which or are had "
    "not but were their one all also new first who has been two more time after other city year most "
    "made into used state may later during would people many some only world over film then school"
).split()


def _trace_replay(params: Mapping[str, Any], seed: int) -> Workload:
    """Replay real Azure LLM trace arrivals and token lengths, scaled down to fit one machine.

    `rate_scale` stretches time (0.1 = one tenth of the trace's request rate); `token_scale` shrinks
    prompt and output lengths, then caps apply. Prompts are filler text of about the target token
    count (the trace has no content), each with a unique header so prefix caching cannot help.
    """
    from maxionbench.datasets.loaders.v03 import load_azure_trace

    unknown = set(params) - TRACE_KEYS
    if unknown:
        raise ValueError(f"trace_replay: unknown params {sorted(unknown)}")
    rate_scale = float(params.get("rate_scale", 1.0))
    token_scale = float(params.get("token_scale", 1.0))
    if rate_scale <= 0 or token_scale <= 0:
        raise ValueError("trace_replay: rate_scale and token_scale must be > 0")
    max_prompt = int(params.get("max_prompt_tokens", 2048))
    max_output = int(params.get("max_output_tokens", 256))
    window = load_azure_trace(float(params.get("start_s", 0.0)), float(params["duration_s"]))
    rng = random.Random(seed)
    specs = []
    for i, (ctx, gen) in enumerate(zip(window.context_tokens.tolist(), window.generated_tokens.tolist())):
        n_prompt = min(max_prompt, max(1, round(ctx * token_scale)))
        words = " ".join(rng.choice(_FILLER) for _ in range(n_prompt))
        specs.append(
            RequestSpec(
                request_id=f"r{i}",
                session_id=f"r{i}",
                prefix_key=f"r{i}",
                messages=({"role": "user", "content": f"Request {i}. Continue this text: {words}"},),
                max_tokens=min(max_output, max(1, round(gen * token_scale))),
            )
        )
    return Workload(
        specs=specs,
        rate_rps=None,
        concurrency=None,
        max_in_flight=int(params.get("max_in_flight", 12)),
        timeout_s=float(params.get("timeout_s", 30.0)),
        max_tokens=max_output,
        warmup_requests=int(params.get("warmup_requests", 0)),
        extra_body={"ignore_eos": True} if params.get("ignore_eos", True) else {},
        arrivals=[t / rate_scale for t in window.arrival_s.tolist()],
    )


def _check_keys(kind: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"{kind}: unknown params {sorted(unknown)}")
    if ("rate_rps" in params) == ("concurrency" in params):
        raise ValueError(f"{kind}: set exactly one of rate_rps (open loop) or concurrency (closed loop)")
