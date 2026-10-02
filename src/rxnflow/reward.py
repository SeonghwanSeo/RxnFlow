"""Injectable, local reward functions."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import QED

SampleFilter = Callable[[Chem.Mol], bool]


class RewardFunction(ABC):
    """Base API for in-process molecular rewards.

    Implementations receive RDKit molecules. Treat them as read-only and use
    ``Chem.MolToSmiles`` if a reward needs strings. ``score`` must return
    finite, non-negative values aligned with the input list.
    """

    @abstractmethod
    def score(self, molecules: list[Chem.Mol]) -> list[float]:
        raise NotImplementedError

    def metrics(self) -> dict[str, float]:
        return {}


class QEDReward(RewardFunction):
    """Example reward using RDKit's quantitative estimate of drug-likeness."""

    def score(self, molecules: list[Chem.Mol]) -> list[float]:
        return [float(QED.qed(mol)) for mol in molecules]


def evaluate_rewards(
    reward: RewardFunction,
    molecules: list[Chem.Mol | None],
    sample_filter: SampleFilter | None = None,
) -> tuple[list[float], dict[str, float]]:
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

    raw_scores = reward.score(accepted)
    if len(raw_scores) != len(accepted):
        raise ValueError("RewardFunction.score returned a misaligned number of rewards")
    scores = [float(value) for value in raw_scores]
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("rewards must be finite and non-negative")
    result = [0.0] * len(molecules)
    for index, score in zip(accepted_indices, scores, strict=True):
        result[index] = score

    metrics = {name: float(value) for name, value in reward.metrics().items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("reward metrics must be finite")
    return result, metrics
