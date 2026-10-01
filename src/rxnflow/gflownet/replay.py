"""Uniform FIFO replay with constant-time ring insertion and indexed sampling."""

from __future__ import annotations

import random

from rxnflow.gflownet.types import Trajectory


class ReplayBuffer:
    def __init__(self, capacity: int):
        assert capacity >= 0
        self.capacity = capacity
        self._items: list[Trajectory] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, trajectories: list[Trajectory]) -> None:
        if not self.capacity:
            return
        for trajectory in trajectories:
            if len(self._items) < self.capacity:
                self._items.append(trajectory)
            else:
                self._items[self._next] = trajectory
                self._next = (self._next + 1) % self.capacity

    def sample(self, count: int, rng: random.Random) -> list[Trajectory]:
        size = len(self._items)
        if count <= 0 or not size:
            return []
        # Logical indices are oldest-first, including after wrap and restart.
        # Draw O(batch size) indices rather than copying the whole replay deque.
        indices = range(size) if count >= size else rng.sample(range(size), count)
        return [self._items[(self._next + i) % size] for i in indices]

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
