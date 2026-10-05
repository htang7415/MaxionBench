"""Versioned harness result model (the dashboard's data contract) and repeat aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import math
from typing import Any, Literal, Mapping, Sequence, get_args, get_origin, get_type_hints

RESULT_SCHEMA_VERSION = "maxionbench-harness-result-v1"

# Two-sided 95% Student-t critical values by degrees of freedom (1..30); normal beyond.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    git_dirty: bool
    spec_fingerprint: str
    started_at: str
    finished_at: str
    host: dict[str, Any]
    tools: dict[str, Any]


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    cell_id: str
    repeat: int
    seed: int
    status: Literal["ok", "failed"]
    started_at: str
    duration_s: float
    host_load_1m_before: float
    quiet_host_ok: bool
    metrics: dict[str, float]
    requests_per_endpoint: list[int]
    target: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class MetricCI:
    mean: float
    ci_low: float
    ci_high: float
    std: float
    n: int


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    params: dict[str, Any]
    n_ok: int
    n_failed: int
    metrics: dict[str, MetricCI]


@dataclass(frozen=True)
class ExperimentResult:
    schema_version: str
    run_id: str
    name: str
    description: str
    spec: dict[str, Any]
    provenance: Provenance
    trials: list[TrialResult]
    cells: list[CellSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mean_ci(values: Sequence[float]) -> MetricCI:
    """Mean with a two-sided 95% Student-t interval (degenerate for n=1)."""
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    mean = sum(values) / n
    if n == 1:
        return MetricCI(mean=mean, ci_low=mean, ci_high=mean, std=0.0, n=1)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = _T95[n - 2] if n - 1 <= len(_T95) else 1.96
    half = t * std / math.sqrt(n)
    return MetricCI(mean=mean, ci_low=mean - half, ci_high=mean + half, std=std, n=n)


def aggregate_cells(trials: Sequence[TrialResult], cell_params: Mapping[str, dict[str, Any]]) -> list[CellSummary]:
    cells: list[CellSummary] = []
    for cell_id, params in cell_params.items():
        ok = [t for t in trials if t.cell_id == cell_id and t.status == "ok"]
        failed = sum(1 for t in trials if t.cell_id == cell_id and t.status != "ok")
        names = sorted({m for t in ok for m in t.metrics})
        metrics = {m: mean_ci([t.metrics[m] for t in ok if m in t.metrics]) for m in names}
        cells.append(CellSummary(cell_id=cell_id, params=params, n_ok=len(ok), n_failed=failed, metrics=metrics))
    return cells


def from_dict(cls: Any, data: Any) -> Any:
    """Strictly rebuild a result dataclass from JSON data; rejects missing or unknown fields."""
    return _coerce(cls, data, cls.__name__)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected object")
        hints = get_type_hints(tp)
        names = {f.name for f in fields(tp)}
        if set(value) != names:
            raise TypeError(f"{where}: fields {sorted(set(value) ^ names)} missing or unexpected")
        return tp(**{n: _coerce(hints[n], value[n], f"{where}.{n}") for n in names})
    if origin is list:
        (item,) = get_args(tp)
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected array")
        return [_coerce(item, v, f"{where}[]") for v in value]
    if origin is dict:
        _, item = get_args(tp)
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected object")
        return {str(k): _coerce(item, v, f"{where}.{k}") for k, v in value.items()}
    if origin is Literal:
        if value not in get_args(tp):
            raise TypeError(f"{where}: {value!r} not in {get_args(tp)}")
        return value
    args = get_args(tp)
    if args and type(None) in args:  # Optional[X]
        if value is None:
            return None
        return _coerce(next(a for a in args if a is not type(None)), value, where)
    if tp is Any:
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected number")
        return float(value)
    if tp in (int, str, bool):
        if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
            raise TypeError(f"{where}: expected {tp.__name__}")
        return value
    raise TypeError(f"{where}: unsupported type {tp!r}")
