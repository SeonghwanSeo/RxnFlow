"""Injectable, local reward functions."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

from rdkit.Chem import QED

from rxnflow.sample import Sample, SampleInput, as_sample

SampleFilter = Callable[[Sample], bool]


class RewardFunction(ABC):
    """Base API for in-process molecular rewards.

    Implementations receive normalized :class:`Sample` objects, which expose
    both canonical ``smiles`` and an RDKit ``mol``. ``score`` must return
    finite, non-negative values aligned with the input list.
    """

    @abstractmethod
    def score(self, samples: list[Sample]) -> list[float]:
        raise NotImplementedError

    def metrics(self) -> dict[str, float]:
        return {}


class QEDReward(RewardFunction):
    """Example reward using RDKit's quantitative estimate of drug-likeness."""

    def score(self, samples: list[Sample]) -> list[float]:
        return [float(QED.qed(sample.mol)) for sample in samples]


def evaluate_rewards(
    reward: RewardFunction,
    values: list[SampleInput],
    sample_filter: SampleFilter | None = None,
) -> tuple[list[float], dict[str, float]]:
    """Normalize samples, filter eligibility, and validate reward output."""

    samples = [as_sample(value) for value in values]
    accepted: list[Sample] = []
    accepted_indices: list[int] = []
    for index, sample in enumerate(samples):
        if sample is None:
            continue
        eligible = True if sample_filter is None else sample_filter(sample)
        if type(eligible) is not bool:
            raise ValueError("sample_filter must return bool")
        if eligible:
            accepted.append(sample)
            accepted_indices.append(index)

    raw_scores = reward.score(accepted)
    if len(raw_scores) != len(accepted):
        raise ValueError("RewardFunction.score returned a misaligned number of rewards")
    scores = [float(value) for value in raw_scores]
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("rewards must be finite and non-negative")
    result = [0.0] * len(values)
    for index, score in zip(accepted_indices, scores, strict=True):
        result[index] = score

    metrics = {name: float(value) for name, value in reward.metrics().items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("reward metrics must be finite")
    return result, metrics
