from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .records import DataSplit


PILOT_TASKS = (
    "CloseSingleDoor",
    "CloseDrawer",
    "TurnOnFaucet",
    "TurnOnMicrowave",
)
TASK_FAMILIES = {
    "CloseSingleDoor": "sustained_articulated_contact",
    "CloseDrawer": "sustained_articulated_contact",
    "TurnOnFaucet": "precise_local_interaction",
    "TurnOnMicrowave": "precise_local_interaction",
    "TurnOnStove": "precise_local_interaction",
}


@dataclass(frozen=True)
class PilotConfig:
    source_count: int = 48
    tasks: tuple[str, ...] = PILOT_TASKS
    layouts: tuple[int, ...] = (1, 4, 7, 8, 9)
    execute_candidates: int = 1
    dock_candidates: int = 5
    assist_candidates: int = 5
    repeats: int = 2
    candidate_seed: int = 1701
    environment_seed_start: int = 10_000
    policy_seed_start: int = 20_000
    collector_batch_size: int = 4

    def validate(self) -> None:
        if self.source_count != 48:
            raise ValueError("MMWAM-OBC-001 pilot requires exactly 48 source states")
        if len(self.tasks) != 4 or len({TASK_FAMILIES[task] for task in self.tasks}) < 2:
            raise ValueError("pilot requires four tasks spanning at least two families")
        if (self.execute_candidates, self.dock_candidates, self.assist_candidates) != (1, 5, 5):
            raise ValueError("pilot requires 1E + 5D + 5A")
        if self.repeats != 2:
            raise ValueError("pilot requires two repeats per candidate")

    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScheduledRollout:
    source_state_id: str
    task_id: str
    task_family: str
    layout_id: int
    stage: str
    collector_batch: int
    route_type: str
    candidate_id: str
    repeat_index: int
    environment_seed: int
    policy_seed: int
    candidate_seed: int
    order_index: int


def _source_rng(seed: int, source_state_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{source_state_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_pilot_schedule(config: PilotConfig = PilotConfig()) -> list[ScheduledRollout]:
    config.validate()
    schedule: list[ScheduledRollout] = []
    for source_index in range(config.source_count):
        source_state_id = f"pilot-source-{source_index:03d}"
        task_id = config.tasks[source_index % len(config.tasks)]
        layout_id = config.layouts[(source_index // len(config.tasks)) % len(config.layouts)]
        candidates = ["e0"]
        candidates.extend(f"d{index}" for index in range(config.dock_candidates))
        candidates.extend(f"a{index}" for index in range(config.assist_candidates))
        branches = [
            (candidate[0].upper(), candidate, repeat)
            for candidate in candidates
            for repeat in range(config.repeats)
        ]
        _source_rng(config.candidate_seed, source_state_id).shuffle(branches)
        for order_index, (route_type, candidate_id, repeat_index) in enumerate(branches):
            schedule.append(
                ScheduledRollout(
                    source_state_id=source_state_id,
                    task_id=task_id,
                    task_family=TASK_FAMILIES[task_id],
                    layout_id=layout_id,
                    stage="precontact",
                    collector_batch=source_index // config.collector_batch_size,
                    route_type=route_type,
                    candidate_id=candidate_id,
                    repeat_index=repeat_index,
                    environment_seed=config.environment_seed_start + source_index * 100 + repeat_index,
                    policy_seed=config.policy_seed_start + source_index * 100 + repeat_index,
                    candidate_seed=config.candidate_seed,
                    order_index=order_index,
                )
            )
    return schedule


def stratified_group_split(
    units: Iterable[tuple[str, ...]], *, seed: int
) -> dict[str, DataSplit]:
    """Assign source IDs within task/layout/stage strata using 60/15/10/15."""
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for unit in units:
        if len(unit) < 2:
            raise ValueError("each unit requires source_state_id and a stratum")
        source_state_id, *stratum = unit
        strata[tuple(stratum)].append(source_state_id)

    fractions = (
        (DataSplit.TRAIN, 0.60),
        (DataSplit.VALIDATION, 0.15),
        (DataSplit.CALIBRATION, 0.10),
        (DataSplit.LOCKED_TEST, 0.15),
    )
    total_units = sum(len(source_ids) for source_ids in strata.values())
    global_raw = [total_units * fraction for _, fraction in fractions]
    global_target = [int(value) for value in global_raw]
    for index in sorted(
        range(len(fractions)),
        key=lambda item: (global_raw[item] - global_target[item], -item),
        reverse=True,
    )[: total_units - sum(global_target)]:
        global_target[index] += 1

    stratum_counts: dict[tuple[str, ...], list[int]] = {}
    remaining_by_split = global_target.copy()
    for stratum, source_ids in sorted(strata.items()):
        counts = [int(len(source_ids) * fraction) for _, fraction in fractions]
        stratum_counts[stratum] = counts
        for index, count in enumerate(counts):
            remaining_by_split[index] -= count

    for stratum, source_ids in sorted(strata.items()):
        counts = stratum_counts[stratum]
        raw = [len(source_ids) * fraction for _, fraction in fractions]
        for _ in range(len(source_ids) - sum(counts)):
            eligible = [index for index, remaining in enumerate(remaining_by_split) if remaining > 0]
            if not eligible:
                raise RuntimeError("global split quota was exhausted early")
            index = max(
                eligible,
                key=lambda item: (
                    raw[item] - counts[item],
                    remaining_by_split[item],
                    -item,
                ),
            )
            counts[index] += 1
            remaining_by_split[index] -= 1

    if any(remaining_by_split):
        raise RuntimeError(f"unallocated global split quotas: {remaining_by_split}")

    result: dict[str, DataSplit] = {}
    for stratum, source_ids in sorted(strata.items()):
        ordered = sorted(
            source_ids,
            key=lambda source_id: hashlib.sha256(
                f"{seed}:{stratum}:{source_id}".encode("utf-8")
            ).digest(),
        )
        counts = stratum_counts[stratum]
        cursor = 0
        for (split, _), count in zip(fractions, counts):
            for source_id in ordered[cursor : cursor + count]:
                if source_id in result:
                    raise ValueError(f"duplicate source state: {source_id}")
                result[source_id] = split
            cursor += count
    return result


def bind_formal_source_count(
    *, feasible_sources: int, minimum: int = 480, maximum: int = 600
) -> int | None:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid formal source-count bounds")
    if feasible_sources < minimum:
        return None
    return min(feasible_sources, maximum)
