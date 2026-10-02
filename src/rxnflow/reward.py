"""Injectable, local reward functions."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch import Tensor

SampleFilter = Callable[[Chem.Mol], bool]


class RewardFunction(ABC):
    """Base API for in-process molecular rewards.

    Implementations receive RDKit molecules. Treat them as read-only and use
    ``Chem.MolToSmiles`` if a reward needs strings. ``score`` must return
    a float32 tensor of shape [batch, len(objectives)], including empty batches.
    Values must be finite, non-negative and larger-is-better. Implementations
    define the objective order and scale; preferences are applied by the trainer.
    """

    objectives: tuple[str, ...]

    @abstractmethod
    def score(self, molecules: list[Chem.Mol]) -> Tensor:
        raise NotImplementedError

    def metrics(self) -> dict[str, float]:
        return {}


class QEDReward(RewardFunction):
    """Example reward using RDKit's quantitative estimate of drug-likeness."""

    objectives = ("qed",)

    def score(self, molecules: list[Chem.Mol]) -> Tensor:
        return torch.tensor(
            [QED.qed(mol) for mol in molecules], dtype=torch.float32
        ).reshape(-1, 1)


def evaluate_rewards(
    reward: RewardFunction,
    molecules: list[Chem.Mol | None],
    sample_filter: SampleFilter | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Filter molecules and align rewards; failed trajectories pass None."""

    accepted: list[Chem.Mol] = []
    accepted_indices: list[int] = []
    for index, mol in enumerate(molecules):
        if mol is None:
            continue
        eligible = True if sample_filter is None else sample_filter(mol)
        if type(eligible) is not bool:
            raise ValueError("sample_filter must return bool")
        if eligible:
            accepted.append(mol)
            accepted_indices.append(index)

    scores = reward.score(accepted).detach()
    if scores.shape != (len(accepted), len(reward.objectives)):
        raise ValueError("RewardFunction.score must return [batch, num_objectives]")
    if scores.dtype != torch.float32:
        raise ValueError("RewardFunction.score must return float32")
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("rewards must be finite and non-negative")
    result = scores.new_zeros((len(molecules), len(reward.objectives)))
    result[accepted_indices] = scores

    metrics = {name: float(value) for name, value in reward.metrics().items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("reward metrics must be finite")
    return result, metrics
