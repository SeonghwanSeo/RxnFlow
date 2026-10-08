"""Numeric condition distributions shared by training and sampling."""

from __future__ import annotations

import math

import torch


def sample_distribution(
    distribution: tuple[str, list[float]], count: int, dim: int = 1
) -> torch.Tensor:
    """Draw [count, dim] using the checkpointed CPU global Torch RNG."""
    name, params = distribution
    if name == "fixed":
        return torch.tensor(params, dtype=torch.float32).expand(count, dim)
    if name == "uniform":
        return torch.empty(count, dim).uniform_(*params)
    if name == "dirichlet":
        # The one-objective simplex is exactly [1], with no random draw.
        if dim == 1:
            return torch.ones(count, 1)
        concentration = torch.tensor(params, dtype=torch.float32).expand(dim)
        return torch.distributions.Dirichlet(concentration).sample((count,))
    raise ValueError(f"unknown distribution: {name}")


class ConditionSampler:
    """Consume parsed distributions; retain sampled numeric labels in replay."""

    def __init__(
        self,
        beta: tuple[str, list[float]],
        preference: tuple[str, list[float]],
        num_objectives: int,
        scalarization: str = "mul",
    ):
        beta_dist, beta_params = beta
        pref_dist, pref_params = preference
        if (
            beta_dist not in ("fixed", "uniform")
            or len(beta_params) != (2 if beta_dist == "uniform" else 1)
            or any(not math.isfinite(x) or x <= 0 for x in beta_params)
            or (beta_dist == "uniform" and beta_params[0] >= beta_params[1])
        ):
            raise ValueError("beta requires positive fixed(value) or uniform(low,high)")
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

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        beta = sample_distribution(self.beta, count).squeeze(-1)
        # Equal relative weights without conditioning; normalize for the reward below.
        preference = (
            torch.ones(count, self.num_objectives)
            if self.preference[0] == "none"
            else sample_distribution(self.preference, count, self.num_objectives)
        )
        preference = preference / preference.sum(-1, keepdim=True) * self.weight_sum
        return beta, preference
