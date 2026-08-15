"""Private bounded trajectory replay."""

from __future__ import annotations

import random
from collections import deque

from rxnflow.types import Trajectory


class ReplayBuffer:
    def __init__(self, capacity: int):
        assert capacity >= 0
        self.capacity = capacity
        self._items: deque[Trajectory] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, trajectories: list[Trajectory]) -> None:
        self._items.extend(trajectories)

    def sample(self, count: int, rng: random.Random) -> list[Trajectory]:
        if count <= 0 or not self._items:
            return []
        items = list(self._items)
        if count >= len(items):
            return items
        return rng.sample(items, count)

    def state_dict(self) -> dict[str, object]:
        return {"capacity": self.capacity, "items": list(self._items)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        items = state["items"]
        assert isinstance(items, list)
        self._items = deque(items, maxlen=self.capacity)
