"""FIFO replay with optional admission limits and uniform replay sampling."""

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
    ):
        assert capacity >= 0
        assert num_insert is None or num_insert >= 0
        assert insert_priority in ("uniform", "reward")
        self.capacity = capacity
        self.num_insert = num_insert
        self.insert_priority = insert_priority
        # Store plain trajectory data, including beta, preference and objective
        # rewards. Sampling preserves these conditions; no relabeling.
        self._items: list[dict[str, Any]] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, trajectories: list[Trajectory], rng: np.random.Generator) -> None:
        if not self.capacity or self.num_insert == 0:
            return
        # Admission acts only on this online batch. Preserve its original order
        # for FIFO eviction, and avoid RNG work when all trajectories are kept.
        if self.num_insert is not None and self.num_insert < len(trajectories):
            if self.insert_priority == "uniform":
                indices = rng.choice(len(trajectories), self.num_insert, replace=False)
            else:
                # Rank by the stored scalar reward under the original preference,
                # before beta. Stable sorting retains input order for tied rewards.
                rewards = np.asarray([t.reward for t in trajectories])
                indices = np.argsort(-rewards, kind="stable")[: self.num_insert]
            trajectories = [trajectories[i] for i in sorted(indices)]
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
        # Logical indices are oldest-first, including after wrap and restart.
        # Draw O(batch size) indices rather than copying the whole replay deque.
        indices = range(size) if count >= size else rng.choice(size, count, replace=False)
        # Reconstruct only the sampled batch. Its molecule objects are not
        # retained by the buffer after the training update finishes.
        return [
            Trajectory.from_dict(self._items[(self._next + i) % size]) for i in indices
        ]

    def state_dict(self) -> dict[str, object]:
        size = len(self._items)
        return {
            "capacity": self.capacity,
            "items": [self._items[(self._next + i) % size] for i in range(size)],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert state["capacity"] == self.capacity
        self._items = list(state["items"])
        self._next = 0
