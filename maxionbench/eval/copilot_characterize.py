"""Step 1: what production coding agents do, from the GitHub Copilot coding-agent traces (June 1-7, 2026).

Measures what context policies and prefix caching act on: context size and growth, prompt composition
by segment type, context drops between consecutive calls (compaction or truncation by the product),
cache hits after drops, model switches and pauses, and the tool and user pauses between calls.

Calls without token accounting (the `tool-model` helper) are left out. A call's `timestamp` marks its
end (start = timestamp - duration): read that way, consecutive calls in a session almost never overlap
(`overlapping_pair_share`), so a session is one sequential chain of calls.

    python -m maxionbench.eval.copilot_characterize [--days 2026-06-01 ...] [--limit 1000] [--jobs 7]
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from maxionbench.datasets.loaders.copilot import DAYS, iter_sessions
from maxionbench.datasets.sources import DATASET_ROOT

DROP_RATIO = 0.7  # next prompt below 70% of the previous one: context was cut, not grown
MIN_DROP_PROMPT = 2_000
GAP_BINS_S = (0, 10, 60, 300, 3600, float("inf"))
GROWTH_INDEXES = (1, 2, 4, 8, 16, 32, 64, 128)
RESULT_SIZE_THRESHOLDS = (2_000, 8_000, 32_000)
CALL_FIELDS = ("prompt", "cached", "completion", "index")  # index: position among the session's calls (1-based)
PAIR_FIELDS = ("gap_s", "ratio", "prev_prompt", "next_cached_frac", "same_turn", "model_switch")


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts[:26].rstrip("Z") + "+00:00").timestamp()


def profile_day(day: str, root: Path = DATASET_ROOT, limit: int | None = None) -> dict[str, Any]:
    calls: dict[str, list[float]] = {k: [] for k in CALL_FIELDS}
    pairs: dict[str, list[float]] = {k: [] for k in PAIR_FIELDS}
    segments: Counter[str] = Counter()
    segments_long: Counter[str] = Counter()  # calls with >= 100k prompt tokens
    models: Counter[str] = Counter()
    model_tokens: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    tool_failures: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    calls_per_turn: list[int] = []
    turns_per_session: list[int] = []
    sessions_with_drop = 0
    batch_ms: list[float] = []
    new_tool_results: list[int] = []  # tokens of each tool result, counted in the call where it first appears
    for session in iter_sessions(day, root, limit):
        counts["sessions"] += 1
        turns = session["turns"] or []
        turns_per_session.append(len(turns))
        chain = []  # (start_s, end_s, turn, model, prompt, cached, completion)
        for t_i, turn in enumerate(turns):
            n_turn = 0
            for c in turn["llm_calls"] or []:
                tok = c["tokens"] or {}
                if not tok.get("prompt"):
                    counts["calls_without_tokens"] += 1
                    continue
                n_turn += 1
                prompt, cached = int(tok["prompt"]), int(tok.get("cached") or 0)
                end = _epoch(c["timestamp"])
                chain.append((end - (c["duration_ms"] or 0) / 1000, end, t_i, c["model"], prompt, cached,
                              int(tok.get("completion") or 0)))
                models[c["model"]] += 1
                model_tokens[c["model"]] += prompt
                seg_total = 0
                for m in c["message_metadata"] or []:
                    key, n = f"{m['type']}/{m['role']}", int(m["token_len"] or 0)
                    segments[key] += n
                    seg_total += n
                    if prompt >= 100_000:
                        segments_long[key] += n
                counts["segment_tokens"] += seg_total
                new_tool_results += _new_tool_results(c["message_metadata"] or [])
                counts["segment_prompt_tokens"] += prompt if seg_total else 0
            calls_per_turn.append(n_turn)
            for b in turn["tool_batches"] or []:
                batch_ms.append(float(b["duration_ms"] or 0))
                for f in b["function_calls"] or []:
                    tools[f["name"]] += 1
                    tool_failures[f["name"]] += f["status"] == 2
        chain.sort(key=lambda call: call[0])
        dropped = False
        for i, (start, end, turn_i, model, prompt, cached, completion) in enumerate(chain):
            calls["prompt"].append(prompt)
            calls["cached"].append(cached)
            calls["completion"].append(completion)
            calls["index"].append(i + 1)
            if i == 0:
                continue
            _, p_end, p_turn, p_model, p_prompt, _, _ = chain[i - 1]
            ratio = prompt / p_prompt
            counts["overlapping_pairs"] += start < p_end - 0.5
            pairs["gap_s"].append(max(0.0, start - p_end))
            pairs["ratio"].append(ratio)
            pairs["prev_prompt"].append(p_prompt)
            pairs["next_cached_frac"].append(cached / prompt)
            pairs["same_turn"].append(turn_i == p_turn)
            pairs["model_switch"].append(model != p_model)
            dropped |= ratio < DROP_RATIO and p_prompt >= MIN_DROP_PROMPT and model == p_model
        sessions_with_drop += dropped
    counts["sessions_with_drop"] = sessions_with_drop
    return {
        "calls": {k: np.asarray(v, dtype=np.float64) for k, v in calls.items()},
        "pairs": {k: np.asarray(v, dtype=np.float64) for k, v in pairs.items()},
        "segments": segments, "segments_long": segments_long, "models": models, "model_tokens": model_tokens,
        "tools": tools, "tool_failures": tool_failures, "counts": counts,
        "calls_per_turn": np.asarray(calls_per_turn), "turns_per_session": np.asarray(turns_per_session),
        "batch_ms": np.asarray(batch_ms), "new_tool_results": np.asarray(new_tool_results),
    }


def _new_tool_results(metadata: list[dict[str, Any]]) -> list[int]:
    """Token lengths of the tool messages after the prompt's last assistant message: the results of the
    tool batch just run. Earlier tool messages were new in an earlier call."""
    ordered = sorted(metadata, key=lambda m: m["sequenceId"])
    last_assistant = max((i for i, m in enumerate(ordered) if m["role"] == "assistant"), default=-1)
    return [int(m["token_len"] or 0) for m in ordered[last_assistant + 1:] if m["role"] == "tool"]


def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in parts[0]:
        if isinstance(parts[0][key], dict) and not isinstance(parts[0][key], Counter):
            out[key] = {k: np.concatenate([p[key][k] for p in parts]) for k in parts[0][key]}
        elif isinstance(parts[0][key], Counter):
            out[key] = sum((p[key] for p in parts), Counter())
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def _pct(values: np.ndarray, qs: tuple[float, ...] = (50, 90, 99)) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {f"p{q:g}": round(float(np.percentile(values, q)), 3) for q in qs} | {"mean": round(float(values.mean()), 3)}


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return round(float(values[mask].mean()), 4) if mask.any() else None


def _shares(counter: Counter[str], top: int | None = None) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 4) for k, v in counter.most_common(top)}


def summarize(d: dict[str, Any]) -> dict[str, Any]:
    c, p, n = d["calls"], d["pairs"], d["counts"]
    results = d["new_tool_results"]
    prompt, cached = c["prompt"], c["cached"]
    order = np.sort(prompt)
    token_weighted_median = float(order[np.searchsorted(np.cumsum(order), order.sum() / 2)])
    same_model = p["model_switch"] == 0
    drop = (p["ratio"] < DROP_RATIO) & (p["prev_prompt"] >= MIN_DROP_PROMPT) & same_model
    steady = same_model & ~drop
    within, between = p["same_turn"] == 1, p["same_turn"] == 0
    gap_bins = {}
    for lo, hi in zip(GAP_BINS_S, GAP_BINS_S[1:]):
        mask = steady & (p["gap_s"] >= lo) & (p["gap_s"] < hi)
        gap_bins[f"{lo:g}-{hi:g}s"] = {"pairs": int(mask.sum()), "cached_frac_mean": _mean(p["next_cached_frac"], mask)}
    tools_top = {name: {"calls": count, "failure_rate": round(d["tool_failures"][name] / count, 4)}
                 for name, count in d["tools"].most_common(20)}
    return {
        "scale": {
            "sessions": n["sessions"], "turns": int(len(d["calls_per_turn"])), "llm_calls": int(len(prompt)),
            "calls_without_tokens": n["calls_without_tokens"], "tool_calls": int(sum(d["tools"].values())),
            "overlapping_pair_share": round(n["overlapping_pairs"] / max(1, len(p["ratio"])), 6),
            "prompt_tokens": int(prompt.sum()), "cached_tokens": int(cached.sum()),
            "completion_tokens": int(c["completion"].sum()),
            "calls_per_turn": _pct(d["calls_per_turn"]), "turns_per_session": _pct(d["turns_per_session"]),
        },
        "context": {
            "prompt_tokens": _pct(prompt) | {"token_weighted_median": token_weighted_median},
            "share_calls_over": {f"{k // 1000}k": round(float((prompt > k).mean()), 4)
                                 for k in (30_000, 64_000, 100_000, 128_000)},
            "median_prompt_by_call_index": {str(i): float(np.median(prompt[c["index"] == i]))
                                            for i in GROWTH_INDEXES if (c["index"] == i).any()},
            "completion_tokens": _pct(c["completion"]),
        },
        "composition": {
            "token_share": _shares(d["segments"]),
            "token_share_prompts_over_100k": _shares(d["segments_long"]),
            "segment_coverage": round(n["segment_tokens"] / max(1, n["segment_prompt_tokens"]), 4),
            "new_tool_result_tokens": _pct(results, (50, 90, 99, 99.9)) | {"count": int(len(results))},
            "tool_result_token_share_from_results_over": {
                f"{k // 1000}k": round(float(results[results > k].sum() / max(1, results.sum())), 4)
                for k in RESULT_SIZE_THRESHOLDS},
        },
        "drops": {
            "definition": f"same model, prompt < {DROP_RATIO} x previous, previous >= {MIN_DROP_PROMPT} tokens",
            "per_1k_pairs": round(1000 * float(drop.mean()), 3),
            "share_sessions": round(n["sessions_with_drop"] / n["sessions"], 4),
            "share_within_turn": _mean(within, drop),
            "prompt_before": _pct(p["prev_prompt"][drop]),
            "ratio": _pct(p["ratio"][drop]),
        },
        "cache": {
            "cached_share": round(float(cached.sum() / prompt.sum()), 4),
            "next_cached_frac_mean": {
                "steady": _mean(p["next_cached_frac"], steady),
                "after_drop": _mean(p["next_cached_frac"], drop),
                "after_model_switch": _mean(p["next_cached_frac"], ~same_model),
                "first_call_of_turn": _mean(p["next_cached_frac"], steady & between),
            },
            "steady_by_gap": gap_bins,
        },
        "pauses_s": {
            "within_turn": _pct(p["gap_s"][within]),
            "between_turns": _pct(p["gap_s"][between]),
            "within_turn_share_over_300s": round(float((p["gap_s"][within] > 300).mean()), 4),
            "tool_batch": _pct(d["batch_ms"] / 1000),
        },
        "models": {"distinct": len(d["models"]), "call_share": _shares(d["models"], 12),
                   "prompt_token_share": _shares(d["model_tokens"], 12),
                   "switch_rate": round(float((~same_model).mean()), 4)},
        "tools": {"top": tools_top, "distinct": len(d["tools"])},
    }


def run(days: list[str], limit: int | None, jobs: int, out_root: Path) -> Path:
    with ProcessPoolExecutor(max_workers=min(jobs, len(days))) as pool:
        parts = list(pool.map(profile_day, days, [DATASET_ROOT] * len(days), [limit] * len(days)))
    summary = {"days": days, "limit_per_day": limit, **summarize(merge(parts))}
    out_dir = out_root / f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-copilot"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Characterize the Copilot coding-agent traces")
    parser.add_argument("--days", nargs="+", default=list(DAYS))
    parser.add_argument("--limit", type=int, help="sessions per day (quick look)")
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/copilot"))
    args = parser.parse_args(argv)
    out_dir = run(args.days, args.limit, args.jobs, args.out)
    print(out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
