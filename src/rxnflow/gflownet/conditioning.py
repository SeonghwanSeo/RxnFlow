"""Numeric condition distributions shared by training and sampling."""

from __future__ import annotations

import math

import numpy as np
import torch


def sample_distribution(
    distribution: tuple[str, list[float]],
    count: int,
    rng: np.random.Generator,
    dim: int = 1,
) -> torch.Tensor:
    """Draw [count, dim] using the supplied checkpointed NumPy RNG."""
    name, params = distribution
    if name == "fixed":
        return torch.tensor(params, dtype=torch.float32).expand(count, dim)
    if name == "uniform":
        values = rng.uniform(*params, size=(count, dim))
        return torch.from_numpy(values.astype(np.float32))
    if name == "log_uniform":
        low, high = params
        values = np.exp(rng.uniform(math.log(low), math.log(high), size=(count, dim)))
        return torch.from_numpy(values.astype(np.float32))
    if name == "dirichlet":
        # The one-objective simplex is exactly [1], with no random draw.
        if dim == 1:
            return torch.ones(count, 1)
        concentration = np.broadcast_to(params, (dim,))
        values = rng.dirichlet(concentration, size=count)
        return torch.from_numpy(values.astype(np.float32))
    raise ValueError(f"unknown distribution: {name}")


class ConditionSampler:
    """Consume parsed distributions; retain sampled numeric labels in replay."""

    def __init__(
        self,
        beta: tuple[str, list[float]],
        preference: tuple[str, list[float]],
        num_objectives: int,
        scalarization: str = "mul",
        *,
        prior_beta: tuple[str, list[float]],
    ):
        pref_dist, pref_params = preference
        self.prior_beta = prior_beta
        if pref_dist == "none":
            if pref_params:
                raise ValueError("none preference takes no parameters")
        elif pref_dist == "fixed":
            if (
                len(pref_params) != num_objectives
                or any(not math.isfinite(x) or x < 0 for x in pref_params)
                or sum(pref_params) <= 0
            ):
                raise ValueError(
                    "fixed preference weights must match objectives "
                    "and have positive total weight"
                )
        elif pref_dist == "dirichlet":
            if len(pref_params) not in (1, num_objectives) or any(
                not math.isfinite(x) or x <= 0 for x in pref_params
            ):
                raise ValueError("Dirichlet preference requires positive concentrations")
        else:
            raise ValueError("preference requires none, fixed or dirichlet distribution")
        self.beta = beta
        self.preference = preference
        self.num_objectives = num_objectives
        self.weight_sum = num_objectives if scalarization == "mul" else 1

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample task beta, preference, then prior beta in checkpoint RNG order."""
        beta = sample_distribution(self.beta, count, rng).squeeze(-1)
        # Equal relative weights without conditioning; normalize for the reward below.
        preference = (
            torch.ones(count, self.num_objectives)
            if self.preference[0] == "none"
            else sample_distribution(self.preference, count, rng, self.num_objectives)
        )
        preference = preference / preference.sum(-1, keepdim=True) * self.weight_sum
        prior_beta = sample_distribution(self.prior_beta, count, rng).squeeze(-1)
        return beta, preference, prior_beta
