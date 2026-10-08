"""Molecular objective evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class RewardFunction(ABC):
    """Abstract base class for reward functions.
    Define ``objectives`` and implement ``score()`` to define a reward function.
    """

    objectives: tuple[str, ...]

    @abstractmethod
    def score(self, smiles_list: list[str]) -> NDArray[np.float32]:
        """Return float32 scores of shape ``(len(smiles_list), len(objectives))``.

        Input: list of unique, uncached SMILES to score.

        Scores must be finite, non-negative and larger-is-better.
        """
        raise NotImplementedError

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @cached_property
    def _cache(self) -> dict[str, NDArray[np.float32]]:
        """Internal cache to avoid re-scoring"""
        return {}

    def __call__(self, smiles_list: list[str | None]) -> NDArray[np.float32]:
        """Evaluate molecules through ``run``."""
        return self.run(smiles_list)

    def run(self, smiles_list: list[str | None]) -> NDArray[np.float32]:
        """Score a batch of molecules with caching."""

        # Create an index before removing None and duplicates
        mapping: dict[str, list[int]] = {}
        for index, smi in enumerate(smiles_list):
            if smi is None:
                continue
            mapping.setdefault(smi, []).append(index)

        # Remove duplicates and cached molecules
        pending = [smi for smi in mapping if smi not in self._cache]

        # Score uncached molecules
        if len(pending) > 0:
            scores = self.score(pending).astype(np.float32)
            # Validate scores
            if scores.shape != (len(pending), len(self.objectives)):
                raise ValueError(
                    "RewardFunction.score must return [batch, num_objectives]"
                )
            if not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError("rewards must be finite and non-negative")

            for smi, values in zip(pending, scores, strict=True):
                self._cache[smi] = values.copy()

        # Fill in the results for all molecules
        result = np.zeros((len(smiles_list), len(self.objectives)), dtype=np.float32)
        for smi, indices in mapping.items():
            result[indices] = self._cache[smi]

        return result
