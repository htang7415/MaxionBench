"""Paired comparison statistics for matched-query audits."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence


def paired_bootstrap(
    a: Mapping[str, float], b: Mapping[str, float], *, resamples: int = 10000, seed: int = 0
) -> dict[str, Any]:
    """Mean of a-b over shared keys with a percentile-bootstrap 95% CI."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no shared keys to pair")
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "n": n,
        "delta": round(sum(diffs) / n, 4),
        "ci95": [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples)], 4)],
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
    }


def paired_generation_deltas(
    records: Sequence[Mapping[str, Any]], comparisons: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """EM/F1 paired deltas between `pipeline@kK` configs from rag_eval generation records."""
    by_config: dict[str, dict[str, Mapping[str, Any]]] = {}
    for rec in records:
        if rec["status"] == "ok":
            by_config.setdefault(f"{rec['pipeline']}@k{rec['k']}", {})[rec["query_id"]] = rec
    out: dict[str, Any] = {}
    for left, right in comparisons:
        if left not in by_config or right not in by_config:
            continue
        out[f"{left} - {right}"] = {
            metric: paired_bootstrap(
                {q: float(r[metric]) for q, r in by_config[left].items()},
                {q: float(r[metric]) for q, r in by_config[right].items()},
            )
            for metric in ("em", "f1")
        }
    return out
