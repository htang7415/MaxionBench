"""Expand an experiment spec into an ordered list of trials."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Any

from maxionbench.harness.spec import ExperimentSpec


@dataclass(frozen=True)
class Trial:
    trial_id: str
    cell_id: str
    repeat: int
    seed: int
    cell_params: dict[str, Any]  # matrix values only, e.g. {"target.routing_policy": "round_robin"}
    target_kind: str
    target_params: dict[str, Any]
    workload_params: dict[str, Any]


def cell_id_for(cell_params: dict[str, Any]) -> str:
    if not cell_params:
        return "default"
    return ",".join(f"{k.split('.', 1)[1]}={v}" for k, v in cell_params.items())


def _order_cells(cells: list[dict[str, Any]], rng: random.Random, *, group_by_target: bool) -> list[dict[str, Any]]:
    """Shuffle cells; when targets are reused, keep cells that share a target contiguous
    (group order and within-group order are both shuffled)."""
    if not group_by_target:
        order = list(cells)
        rng.shuffle(order)
        return order
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        key = repr(sorted((k, repr(v)) for k, v in cell.items() if k.startswith("target.")))
        groups.setdefault(key, []).append(cell)
    keys = list(groups)
    rng.shuffle(keys)
    order = []
    for key in keys:
        members = list(groups[key])
        rng.shuffle(members)
        order.extend(members)
    return order


def plan(spec: ExperimentSpec) -> list[Trial]:
    """Repeat-major order; cell order is reshuffled per repeat so host drift spreads across cells."""
    axes = list(spec.matrix.items())
    cells = [dict(zip([k for k, _ in axes], combo)) for combo in itertools.product(*[v for _, v in axes])] or [{}]
    rng = random.Random(spec.seed)
    trials: list[Trial] = []
    for repeat in range(spec.repeats):
        order = _order_cells(cells, rng, group_by_target=spec.reuse_targets)
        seed = spec.seed + repeat if spec.seed_strategy == "per_repeat" else spec.seed
        for cell in order:
            target = spec.target_variants[str(cell["target.variant"])] if "target.variant" in cell else spec.target
            target_params = dict(target.params)
            workload_params = dict(spec.workload.params)
            for key, value in cell.items():
                if key == "target.variant":
                    continue
                section, _, param = key.partition(".")
                (target_params if section == "target" else workload_params)[param] = value
            cid = cell_id_for(cell)
            trials.append(
                Trial(
                    trial_id=f"r{repeat}-{cid}",
                    cell_id=cid,
                    repeat=repeat,
                    seed=seed,
                    cell_params=cell,
                    target_kind=target.kind,
                    target_params=target_params,
                    workload_params=workload_params,
                )
            )
    return trials
