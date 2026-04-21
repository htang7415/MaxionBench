"""Robustness metrics."""

from __future__ import annotations


def p99_inflation(current_p99_ms: float, baseline_p99_ms: float) -> float:
    if baseline_p99_ms < 0:
        raise ValueError("baseline_p99_ms must be >= 0")
    if baseline_p99_ms == 0:
        return 0.0
    return current_p99_ms / baseline_p99_ms


def sla_violation_rate(total_requests: int, over_sla: int, errors: int) -> float:
    if total_requests <= 0:
        return 0.0
    return (over_sla + errors) / total_requests
