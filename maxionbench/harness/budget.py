"""Hard spend cap for paid APIs: price table, cost model, and a persistent reserve/commit ledger.

A run must `reserve` its estimated cost before sending any request; the ledger refuses if committed
spend + open reservations + estimate would exceed the cap. Actual cost is committed from provider
usage afterwards. The ledger lives outside the repo (default ~/.maxionbench/budget) so cleaning
artifacts never resets spend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping
import uuid

import yaml

from maxionbench.schemas.result_schema import utc_now_iso

DEFAULT_PRICING_PATH = Path("configs/pricing/gemini.yaml")
BUDGET_DIR_ENV = "MAXIONBENCH_BUDGET_DIR"


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float
    cache_storage_per_m_hour: float
    batch_input_per_m: float
    batch_output_per_m: float


@dataclass(frozen=True)
class PriceTable:
    source: str
    retrieved: str
    budget_cap_usd: float
    models: dict[str, ModelPrice]

    def price(self, model: str) -> ModelPrice:
        if model not in self.models:
            raise KeyError(f"no price for {model!r}; add it to the pricing config before spending")
        return self.models[model]


def load_prices(path: Path = DEFAULT_PRICING_PATH) -> PriceTable:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PriceTable(
        source=str(raw["source"]),
        retrieved=str(raw["retrieved"]),
        budget_cap_usd=float(raw["budget_cap_usd"]),
        models={name: ModelPrice(**{k: float(v) for k, v in p.items()}) for name, p in raw["models"].items()},
    )


def cost_usd(
    price: ModelPrice, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0, batch: bool = False
) -> float:
    """Cached tokens are a subset of input tokens and are billed at the cached rate."""
    if min(input_tokens, output_tokens, cached_tokens) < 0 or cached_tokens > input_tokens:
        raise ValueError("token counts must be >= 0 and cached_tokens <= input_tokens")
    input_rate = price.batch_input_per_m if batch else price.input_per_m
    output_rate = price.batch_output_per_m if batch else price.output_per_m
    return (
        (input_tokens - cached_tokens) * input_rate
        + cached_tokens * price.cached_input_per_m
        + output_tokens * output_rate
    ) / 1_000_000


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    label: str
    estimate_usd: float


class BudgetLedger:
    """Append-only JSONL ledger of reserve/commit/release events.

    Exposure = committed actual cost + open reservations. Every check-and-append runs under an
    exclusive file lock, so the cap holds across threads and processes.
    """

    def __init__(self, cap_usd: float, path: Path | None = None) -> None:
        if cap_usd <= 0:
            raise ValueError("cap_usd must be > 0")
        default_dir = Path(os.environ.get(BUDGET_DIR_ENV) or Path.home() / ".maxionbench" / "budget")
        self.path = Path(path) if path else default_dir / "gemini_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()

    def spent_usd(self) -> float:
        with self._locked(fcntl.LOCK_SH) as fh:
            committed, _ = _tally(fh)
        return committed

    def remaining_usd(self) -> float:
        with self._locked(fcntl.LOCK_SH) as fh:
            committed, open_ = _tally(fh)
        return self.cap_usd - committed - sum(open_.values())

    def reserve(self, estimate_usd: float, label: str) -> Reservation:
        if estimate_usd < 0:
            raise ValueError("estimate_usd must be >= 0")
        with self._locked(fcntl.LOCK_EX) as fh:
            committed, open_ = _tally(fh)
            remaining = self.cap_usd - committed - sum(open_.values())
            if estimate_usd > remaining:
                raise BudgetExceededError(
                    f"{label}: estimate ${estimate_usd:.4f} exceeds remaining ${remaining:.4f} "
                    f"of ${self.cap_usd:.2f} cap"
                )
            reservation = Reservation(uuid.uuid4().hex, label, estimate_usd)
            _append(fh, {"event": "reserve", "reservation_id": reservation.reservation_id, "label": label,
                         "estimate_usd": round(estimate_usd, 6)})
            return reservation

    def commit(self, reservation: Reservation, actual_usd: float, usage: Mapping[str, Any]) -> None:
        if actual_usd < 0:
            raise ValueError("actual_usd must be >= 0")
        with self._locked(fcntl.LOCK_EX) as fh:
            _append(fh, {"event": "commit", "reservation_id": reservation.reservation_id, "label": reservation.label,
                         "actual_usd": round(actual_usd, 6), "usage": dict(usage)})

    def release(self, reservation: Reservation) -> None:
        """Close a reservation that spent nothing (e.g. the run failed before its first request)."""
        with self._locked(fcntl.LOCK_EX) as fh:
            _append(fh, {"event": "release", "reservation_id": reservation.reservation_id})

    @contextmanager
    def _locked(self, mode: int) -> Iterator[Any]:
        with self._lock, self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh, mode)
            fh.seek(0)
            yield fh


def _tally(fh: Any) -> tuple[float, dict[str, float]]:
    committed = 0.0
    open_: dict[str, float] = {}
    for line in fh:
        if not line.strip():
            continue
        event = json.loads(line)
        rid = event["reservation_id"]
        if event["event"] == "reserve":
            open_[rid] = float(event["estimate_usd"])
        elif event["event"] == "commit":
            committed += float(event["actual_usd"])
            open_.pop(rid, None)
        elif event["event"] == "release":
            open_.pop(rid, None)
    return committed, open_


def _append(fh: Any, event: dict[str, Any]) -> None:
    fh.seek(0, os.SEEK_END)
    fh.write(json.dumps({"at": utc_now_iso(), **event}) + "\n")
    fh.flush()
