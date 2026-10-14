"""Sampling and fixed feature encoding of external conditions."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

_ConditionT = TypeVar("_ConditionT", bound=dict[str, Any])


class ConditioningModule(ABC, Generic[_ConditionT]):
    """External condition module"""

    encoding_dim: int

    @abstractmethod
    def sample(self, num_samples: int, rng: np.random.Generator) -> list[_ConditionT]:
        """Return serializable raw conditions to retain with each trajectory."""

    @abstractmethod
    def encode(self, cond_list: list[_ConditionT]) -> NDArray[np.float32]:
        """Return deterministic float32 features [len(cond_list), encoding_dim]."""
