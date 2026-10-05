"""Client-side endpoint pickers for multi-replica LLM serving.

`PrefixAffinity` mirrors the idea behind llm-d's endpoint picker: send requests that share a
prompt prefix to the same replica so its KV/prefix cache is reused, unless that replica is
overloaded relative to the least-loaded healthy one.
"""

from __future__ import annotations

import hashlib
import threading


class EndpointPicker:
    """Base picker: tracks in-flight requests and passive health per endpoint."""

    name = "base"

    def __init__(self, num_endpoints: int) -> None:
        if num_endpoints < 1:
            raise ValueError("num_endpoints must be >= 1")
        self._lock = threading.Lock()
        self.outstanding = [0] * num_endpoints
        self.healthy = [True] * num_endpoints

    def acquire(self, prefix_key: str) -> int | None:
        """Pick an endpoint and count the request as in flight; None if no endpoint is healthy."""
        with self._lock:
            candidates = [i for i, ok in enumerate(self.healthy) if ok]
            if not candidates:
                return None
            idx = self._choose(prefix_key, candidates)
            self.outstanding[idx] += 1
            return idx

    def release(self, idx: int) -> None:
        with self._lock:
            self.outstanding[idx] -= 1

    def mark_down(self, idx: int) -> None:
        with self._lock:
            self.healthy[idx] = False

    def _least_loaded(self, candidates: list[int]) -> int:
        return min(candidates, key=lambda i: (self.outstanding[i], i))

    def _choose(self, prefix_key: str, candidates: list[int]) -> int:
        raise NotImplementedError


class RoundRobin(EndpointPicker):
    name = "round_robin"

    def __init__(self, num_endpoints: int) -> None:
        super().__init__(num_endpoints)
        self._next = 0

    def _choose(self, prefix_key: str, candidates: list[int]) -> int:
        idx = candidates[self._next % len(candidates)]
        self._next += 1
        return idx


class LeastOutstanding(EndpointPicker):
    name = "least_outstanding"

    def _choose(self, prefix_key: str, candidates: list[int]) -> int:
        return self._least_loaded(candidates)


class PrefixAffinity(EndpointPicker):
    name = "prefix_affinity"

    def __init__(self, num_endpoints: int, max_imbalance: int = 2) -> None:
        super().__init__(num_endpoints)
        self.max_imbalance = max_imbalance

    def _choose(self, prefix_key: str, candidates: list[int]) -> int:
        # Rendezvous hashing: losing a replica only remaps the keys that lived on it.
        home = max(candidates, key=lambda i: _hash64(f"{prefix_key}|{i}"))
        least = self._least_loaded(candidates)
        if self.outstanding[home] - self.outstanding[least] > self.max_imbalance:
            return least
        return home


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


PICKERS: dict[str, type[EndpointPicker]] = {
    cls.name: cls for cls in (RoundRobin, LeastOutstanding, PrefixAffinity)
}
