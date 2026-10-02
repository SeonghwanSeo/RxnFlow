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
        preferences: tuple[str, list[float]],
        num_objectives: int,
    ):
        beta_dist, beta_params = beta
        pref_dist, pref_params = preferences
        if (
            beta_dist not in ("fixed", "uniform")
            or len(beta_params) != (2 if beta_dist == "uniform" else 1)
            or any(not math.isfinite(x) or x <= 0 for x in beta_params)
            or (beta_dist == "uniform" and beta_params[0] >= beta_params[1])
        ):
            raise ValueError("beta requires positive fixed(value) or uniform(low,high)")
        if pref_dist == "fixed":
            if (
                len(pref_params) != num_objectives
                or any(not math.isfinite(x) or x < 0 for x in pref_params)
                or not math.isclose(sum(pref_params), 1.0, abs_tol=1e-6)
            ):
                raise ValueError("fixed preferences must match objectives and sum to 1")
        elif pref_dist == "dirichlet":
            if len(pref_params) not in (1, num_objectives) or any(
                not math.isfinite(x) or x <= 0 for x in pref_params
            ):
                raise ValueError("Dirichlet preferences require positive concentrations")
        else:
            raise ValueError("preferences require fixed or dirichlet distribution")
        self.beta = beta
        self.preferences = preferences
        self.num_objectives = num_objectives

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        beta = sample_distribution(self.beta, count).squeeze(-1)
        preferences = sample_distribution(self.preferences, count, self.num_objectives)
        return beta, preferences
