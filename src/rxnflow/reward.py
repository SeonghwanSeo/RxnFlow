"""Reward function definition"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RewardFunction(ABC):
    """Abstract base class for reward functions.
    Define ``objectives`` and implement ``score()`` to define a reward function.
    """

    objectives: tuple[str, ...]

    @abstractmethod
    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Return float32 scores of shape ``(len(smiles_list), len(objectives))``.

        Without conditions, inputs are unique, uncached SMILES. With conditions,
        molecules and conditions retain their matching order, including duplicates.

        Scores must be finite, non-negative and larger-is-better.
        """
        raise NotImplementedError

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @cached_property
    def _cache(self) -> dict[str, NDArray[np.float32]]:
        """Cache scores by SMILES for calls without conditions."""
        return {}

    def __call__(
        self,
        smiles_list: list[str | None],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Evaluate molecules through ``run``."""
        return self.run(smiles_list, cond_list)

    def run(
        self,
        smiles_list: list[str | None],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Score a batch of molecules."""
        num_samples = len(smiles_list)

        # Initialize the result array with zeros
        result = np.zeros((num_samples, self.num_objectives), dtype=np.float32)

        if cond_list is not None:
            # If conditions are provided, skip caching or deduplication
            assert len(cond_list) == len(smiles_list), (
                "Length of cond_list must match length of smiles_list"
            )
            indices = [i for i, smi in enumerate(smiles_list) if smi is not None]
            if not indices:
                return result
            _smiles_list = [smiles_list[i] for i in indices]
            _cond_list = [cond_list[i] for i in indices]
            scores = self.score(_smiles_list, _cond_list).astype(np.float32)
            if scores.shape != (len(_smiles_list), len(self.objectives)):
                raise ValueError(
                    "RewardFunction.score must return [batch, num_objectives]"
                )
            if not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError("rewards must be finite and non-negative")
            result[indices] = scores
            return result

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
        for smi, indices in mapping.items():
            result[indices] = self._cache[smi]

        return result
