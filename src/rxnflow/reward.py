"""Molecular objective evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem


class RewardFunction(ABC):
    """Abstract base class for reward functions.
    Define ``objectives`` and implement ``score()`` to define a reward function.
    """

    objectives: tuple[str, ...]

    @abstractmethod
    def score(self, mols: list[Chem.Mol]) -> NDArray[np.float32]:
        """Return float32 scores of shape ``(len(mols), len(objectives))``.

        Scores must be finite, non-negative and larger-is-better. Preserve input
        order, support empty batches, and do not modify the input molecules.
        """
        raise NotImplementedError

    def __call__(self, mols: list[Chem.Mol | None]) -> NDArray[np.float32]:
        """Evaluate molecules through ``run``."""
        return self.run(mols)

    def run(self, mols: list[Chem.Mol | None]) -> NDArray[np.float32]:
        """Validate objective scores and return them in input order.

        ``None`` entries receive zero scores and are not passed to ``score``.
        """

        # 1. Collect valid molecules, retaining positions in the original batch.
        accepted: list[Chem.Mol] = []
        accepted_indices: list[int] = []
        for index, mol in enumerate(mols):
            if mol is None:
                continue
            accepted.append(mol)
            accepted_indices.append(index)

        # 2. Score accepted molecules once, including an empty accepted batch.
        scores = self.score(accepted)
        if scores.shape != (len(accepted), len(self.objectives)):
            raise ValueError("RewardFunction.score must return [batch, num_objectives]")
        if scores.dtype != np.float32:
            raise ValueError("RewardFunction.score must return float32")
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("rewards must be finite and non-negative")
        # 3. Scatter objectives back; invalid trajectories keep zero reward.
        result = np.zeros((len(mols), len(self.objectives)), dtype=np.float32)
        result[accepted_indices] = scores

        return result
