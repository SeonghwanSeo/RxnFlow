"""FIFO replay with optional admission limits and uniform or reward-weighted sampling."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from rxnflow.core.types import Trajectory


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_insert: int | None = None,
        insert_priority: Literal["uniform", "reward"] = "uniform",
        sample_priority: Literal["uniform", "reward"] = "uniform",
    ):
        assert capacity >= 0
        assert num_insert is None or num_insert >= 0
        assert insert_priority in ("uniform", "reward")
        assert sample_priority in ("uniform", "reward")

        self.capacity: int = capacity
        self.num_insert: int | None = num_insert
        self.insert_priority: str = insert_priority
        self.sample_priority: str = sample_priority

        # Buffer
        self._items: list[dict[str, Any]] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, trajectories: list[Trajectory], rng: np.random.Generator) -> None:
        if not self.capacity or self.num_insert == 0:
            return

        # Select trajectories to insert
        if self.num_insert is not None and self.num_insert < len(trajectories):
            if self.insert_priority == "uniform":
                indices = rng.choice(len(trajectories), self.num_insert, replace=False)
            else:
                # Rank by the stored scalar reward under the original preference,
                # before beta. Stable sorting retains input order for tied rewards.
                rewards = np.asarray([t.reward for t in trajectories])
                indices = np.argsort(-rewards, kind="stable")[: self.num_insert]
            trajectories = [trajectories[i] for i in sorted(indices)]

        # Insert trajectories into the buffer
        for trajectory in trajectories:
            item = trajectory.to_dict()
            if len(self._items) < self.capacity:
                self._items.append(item)
            else:
                self._items[self._next] = item
                self._next = (self._next + 1) % self.capacity

    def sample(self, count: int, rng: np.random.Generator) -> list[Trajectory]:
        size = len(self._items)
        if count <= 0 or not size:
            return []

        # Select trajectories to sample
        if count >= size:
            indices = list(range(size))
        else:
            if self.sample_priority == "uniform":
                indices = rng.choice(size, count, replace=False)
            else:
                weights = np.asarray([t["reward"] for t in self._items]).clip(min=1e-8)
                p = weights / weights.sum()
                indices = rng.choice(size, count, replace=False, p=p)

        return [Trajectory.from_dict(self._items[i]) for i in indices]

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "items": self._items.copy(),
            "next": self._next,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert state["capacity"] == self.capacity
        self._items = list(state["items"])
        # TODO: Require the saved next slot in the next version.
        # Older checkpoints store items oldest-first, so slot zero is next.
        self._next = state.get("next", 0)
