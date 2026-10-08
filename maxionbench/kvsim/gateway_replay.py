"""K9: Copilot agent sessions replayed as chat requests through the Go gateway's context manager onto a real engine.

K6-K8 replayed precomputed per-policy traces as token-id prompts. Here the client sends what a real agent
sends, its full message history every call, and the gateway (gateway/internal/ctxmgr) decides the view, so
the measured system is the shipped mechanism. Each Copilot call's message metadata (role, token count)
becomes a message whose text is deterministic filler of `scale` x its token count (one short word per
token), keyed by the message, so an unchanged message is the same text in every call and the engine's
prefix cache sees exactly the prefix reuse the trace implies. Copilot's own history rewrites pass through
unchanged and reset the gateway's session (it sees the history diverge).

    python -m maxionbench.kvsim.gateway_replay experiments/k9_gateway_context_8b.yaml [--out results]

Open loop like kvsim.live: arrivals follow the trace (gaps idle-capped, then divided by `time_scale`).
"""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlsplit

import yaml

from maxionbench.datasets.loaders.copilot import iter_archive
from maxionbench.datasets.sources import verified_path
from maxionbench.harness.gateway import AIGateway, scrape_gateway_metrics
from maxionbench.harness.provenance import make_provenance
from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult, TrialResult, aggregate_cells
from maxionbench.harness.targets import scrape_vllm_counters
from maxionbench.kvsim.copilot import _end_s, calls_of, history, sample_ids
from maxionbench.kvsim.live import Outcome, schedule, trial_metrics
from maxionbench.schemas.result_schema import utc_now_iso

SPEC_SCHEMA = "maxionbench-gwreplay-v1"
CONTEXT_HEADER = "X-Maxionbench-Context"
# Common three-letter words: " the" etc. are one token each in the Qwen3 tokenizer, so text has ~1 token
# per word and ~4 characters per token, which is also what the gateway's estimator assumes.
WORDS = ("the and for are but not you all any can had her was one our out day get has him his how man new now "
         "old see two way who boy did its let put say she too use").split()


@dataclass(frozen=True)
class ChatCall:
    t: float  # arrival, seconds from session start (idle-capped)
    dur: float
    messages: tuple[tuple[str, str, int], ...]  # (role, key, full-scale tokens)
    out_tokens: int


@dataclass(frozen=True)
class ChatSession:
    id: str
    requests: tuple[ChatCall, ...]
    span: float


@dataclass
class ChatOutcome(Outcome):
    action: str = ""


def session_calls(session: dict[str, Any], idle_cap_s: float) -> ChatSession:
    """A Copilot session's calls as message lists, timed from the first call's start with idle gaps capped."""
    first_len: dict[str, int] = {}
    timed = []
    for c in calls_of(session):
        dur = float(c["duration_ms"] or 0) / 1000
        # per-message counts drift between calls for unchanged messages: keep the first one (as kvsim.copilot)
        msgs = tuple((m["role"], m["content"], first_len.setdefault(m["content"], m["tokens"])) for m in history(c))
        timed.append((_end_s(c["timestamp"]) - dur, dur, msgs, int(c["tokens"].get("completion") or 0)))
    calls, shift, busy_until = [], 0.0, timed[0][0]
    for start, dur, msgs, out in timed:
        if start - busy_until > idle_cap_s:
            shift += start - busy_until - idle_cap_s
        busy_until = max(busy_until, start + dur)
        calls.append(ChatCall(t=start - shift - timed[0][0], dur=dur, messages=msgs, out_tokens=out))
    return ChatSession(id=session["session_id"], requests=tuple(calls), span=max(c.t + c.dur for c in calls))


def load_chat_sessions(archive: Path, sessions: int, seed: int, idle_cap_s: float, cache: Path) -> list[ChatSession]:
    """Seeded sessions of one Copilot day (cached locally; the derivation is deterministic)."""
    if not cache.exists():
        ids = sample_ids(archive, sessions, seed)
        rows = [s for s in iter_archive(archive) if s["session_id"] in ids]
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        tmp.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(cache)
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]
    return sorted((session_calls(r, idle_cap_s) for r in rows), key=lambda s: s.id)


class Renderer:
    """Deterministic filler text per (session, message key), `scale` x the message's tokens long."""

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self._text: dict[tuple[str, str], str] = {}

    def text(self, sid: str, key: str, tokens: int) -> str:
        k = (sid, key)
        if k not in self._text:
            seed = int.from_bytes(hashlib.sha256(f"{sid}|{key}".encode()).digest()[:8], "big")
            rng = random.Random(seed)
            self._text[k] = " ".join(rng.choice(WORDS) for _ in range(max(1, round(tokens * self.scale))))
        return self._text[k]

    def messages(self, sid: str, call: ChatCall) -> list[dict[str, Any]]:
        out = []
        for role, key, tokens in call.messages:
            m: dict[str, Any] = {"role": role, "content": self.text(sid, key, tokens)}
            if role == "tool":
                m["tool_call_id"] = hashlib.sha256(key.encode()).hexdigest()[:12]
            out.append(m)
        return out


def stream_chat(base_url: str, body: dict[str, Any], timeout_s: float) -> tuple[float | None, dict[str, Any], str]:
    """POST a streaming chat completion; (seconds to the first content chunk, usage, context action)."""
    parts = urlsplit(base_url)
    conn = http.client.HTTPConnection(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    started = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"content-type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"http {resp.status}: {resp.read(300).decode('utf-8', 'replace')}")
        action = resp.getheader(CONTEXT_HEADER) or "off"
        ttft, usage = None, {}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if ttft is None and any((c.get("delta") or {}).get("content") for c in chunk.get("choices") or []):
                ttft = time.perf_counter() - started
            usage = chunk.get("usage") or usage
        return ttft, usage, action
    finally:
        conn.close()


def replay(arrivals: Sequence[Any], sessions: Sequence[ChatSession], base_url: str, render: Renderer, *,
           output_scale: float, max_output_tokens: int, timeout_s: float) -> list[ChatOutcome]:
    outcomes: list[ChatOutcome | None] = [None] * len(arrivals)
    t0 = time.perf_counter()

    def send(i: int, a: Any) -> None:
        s = sessions[a.session]
        try:
            body = {"model": "local", "messages": render.messages(s.id, a.request), "prompt_cache_key": s.id,
                    "max_tokens": max(1, min(max_output_tokens, round(a.request.out_tokens * output_scale))),
                    "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            sent = time.perf_counter()
            ttft, usage, action = stream_chat(base_url, body, timeout_s)
            done = time.perf_counter()
            queued = sent - (t0 + a.t)
            outcomes[i] = ChatOutcome(
                scheduled_s=a.t, status="ok", ttft_s=None if ttft is None else queued + ttft, e2e_s=done - t0 - a.t,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)), action=action)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, none stops the replay
            outcomes[i] = ChatOutcome(scheduled_s=a.t, status="error", error=str(exc)[:200])

    with ThreadPoolExecutor(max_workers=256) as pool:
        for i, a in enumerate(arrivals):
            delay = t0 + a.t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(send, i, a)
    return [o for o in outcomes if o is not None]


def action_shares(outcomes: Sequence[ChatOutcome], warmup_s: float) -> dict[str, float]:
    ok = [o for o in outcomes if o.status == "ok" and o.scheduled_s >= warmup_s]
    return {f"context_{a}_share": sum(o.action == a for o in ok) / len(ok)
            for a in ("append", "edit", "pause_edit", "start")} if ok else {}


def gateway_params(spec: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    return {"local": {"kind": "vllm_metal", "params": dict(spec["target"])}, "policy": "local_only",
            "max_inflight": 1_000, "local_model": spec["target"]["model"],
            "port": int(spec.get("gateway_port", 8090)), "context": dict(arm)}


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SPEC_SCHEMA}")
    for key in ("name", "days", "arms", "replay", "target"):
        if key not in spec:
            raise ValueError(f"{path}: missing {key}")
    return spec


def run(spec_path: Path, out_root: Path, log=lambda m: print(m, file=sys.stderr)) -> Path:
    spec = load_spec(spec_path)
    rp = spec["replay"]
    started_at = utc_now_iso()
    n, seed0, idle_cap = int(spec.get("sessions", 200)), int(spec.get("seed", 0)), float(spec.get("idle_cap_s", 300))
    cache_dir = Path(spec.get("cache_dir", "artifacts/cache/k9"))
    sessions = {day: load_chat_sessions(verified_path(rel), n, seed0, idle_cap,
                                        cache_dir / f"{Path(rel).name.split('.tar')[0]}.n{n}.s{seed0}.jsonl")
                for day, rel in spec["days"].items()}
    render = Renderer(float(rp["scale"]))
    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{spec['name']}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    cells = {f"day={day}/context={arm}": {"day": day, "arm": arm} for day in spec["days"] for arm in spec["arms"]}
    warmup = float(rp["warmup_s"])
    trials: list[TrialResult] = []
    for rep in range(int(spec.get("repeats", 1))):
        seed = seed0 * 1000 + rep
        arrivals = {day: schedule(s, int(rp["concurrency"]), float(rp["horizon_s"]), warmup,
                                  float(rp.get("time_scale", 1.0)), seed, mid_life=bool(rp.get("mid_life", False)))
                    for day, s in sessions.items()}
        for cell_id, cell in cells.items():  # every arm replays the same session schedule within a day and repeat
            label = f"{cell_id}/r{rep}"
            log(f"{label}: {len(arrivals[cell['day']])} arrivals over {rp['horizon_s']} s")
            t_start, trial_started = time.perf_counter(), utc_now_iso()
            gw = AIGateway(gateway_params(spec, spec["arms"][cell["arm"]]), out_dir / "logs" / label.replace("/", "_"))
            with gw:
                outcomes = replay(arrivals[cell["day"]], sessions[cell["day"]], gw.base_urls[0], render,
                                  output_scale=float(rp["output_scale"]), max_output_tokens=int(rp["max_output_tokens"]),
                                  timeout_s=float(rp["timeout_s"]))
                engine = scrape_vllm_counters(gw.inner.base_urls[0])
                gateway = scrape_gateway_metrics(gw.base_urls[0])
            metrics = trial_metrics(outcomes, warmup, float(rp["ttft_slo_s"]))
            metrics.update(action_shares(outcomes, warmup))
            queries = engine.get("prefix_cache_queries_total", 0.0)
            metrics["engine_prefix_hit_ratio"] = engine.get("prefix_cache_hits_total", 0.0) / queries if queries else 0.0
            log(f"{label}: " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            trials.append(TrialResult(
                trial_id=label, cell_id=cell_id, repeat=rep, seed=seed, status="ok", started_at=trial_started,
                duration_s=round(time.perf_counter() - t_start, 3), host_load_1m_before=0.0, quiet_host_ok=True,
                metrics={k: round(v, 6) for k, v in metrics.items()}, requests_per_endpoint=[len(outcomes)],
                target={**gw.describe(), "gateway_metrics": gateway}, error=None))
            with (out_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
                for o in outcomes:
                    fh.write(json.dumps({"trial": label, **o.__dict__}) + "\n")
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION, run_id=run_id, name=spec["name"], description=spec.get("description", ""),
        spec=spec, provenance=make_provenance(spec, started_at, {"days": spec["days"], "trials_completed": len(trials)}),
        trials=trials, cells=aggregate_cells(trials, cells))
    (out_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="python -m maxionbench.kvsim.gateway_replay", description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    run(args.spec, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
