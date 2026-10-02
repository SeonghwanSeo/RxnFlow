"""Injectable, local reward functions."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem


class RewardFunction(ABC):
    """Base API for in-process molecular rewards.

    Implementations receive RDKit molecules. Treat them as read-only and use
    ``Chem.MolToSmiles`` if a reward needs strings. ``score`` must return
    a float32 NumPy array of shape [batch, len(objectives)], including empty batches.
    Values must be finite, non-negative and larger-is-better. Implementations
    define the objective order and scale; preferences are applied by the trainer.
    """

    objectives: tuple[str, ...]

    @abstractmethod
    def score(self, molecules: list[Chem.Mol]) -> NDArray[np.float32]:
        raise NotImplementedError

    def filter_object(self, mol: Chem.Mol) -> bool:
        """Return False to skip scoring and assign zero objective rewards."""
        return True

    def metrics(self) -> dict[str, float]:
        return {}


def evaluate_rewards(
    reward: RewardFunction,
    molecules: list[Chem.Mol | None],
) -> tuple[NDArray[np.float32], dict[str, float]]:
    """Filter molecules and align rewards; failed trajectories pass None."""

    # 1. Filter valid molecules, retaining positions in the original batch.
    accepted: list[Chem.Mol] = []
    accepted_indices: list[int] = []
    for index, mol in enumerate(molecules):
        if mol is None:
            continue
        eligible = reward.filter_object(mol)
        if type(eligible) is not bool:
            raise ValueError("RewardFunction.filter_object must return bool")
        if eligible:
            accepted.append(mol)
            accepted_indices.append(index)

    # 2. Score accepted molecules once, including an empty accepted batch.
    scores = reward.score(accepted)
    if scores.shape != (len(accepted), len(reward.objectives)):
        raise ValueError("RewardFunction.score must return [batch, num_objectives]")
    if scores.dtype != np.float32:
        raise ValueError("RewardFunction.score must return float32")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("rewards must be finite and non-negative")
    # 3. Scatter objectives back; invalid and filtered molecules keep zero reward.
    result = np.zeros((len(molecules), len(reward.objectives)), dtype=np.float32)
    result[accepted_indices] = scores

    metrics = {name: float(value) for name, value in reward.metrics().items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("reward metrics must be finite")
    return result, metrics
