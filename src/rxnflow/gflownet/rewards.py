"""Combine molecular objectives into GFlowNet rewards."""

import torch


def scalarize_log_rewards(
    values: torch.Tensor, preferences: torch.Tensor, method: str, floor: float
) -> torch.Tensor:
    """Return the log of the weighted product or sum, before applying beta.

    Weights must sum to the number of objectives for ``mul`` or to one for
    ``sum``. Apply the floor to each objective for ``mul``, or to the sum.
    """
    if method == "mul":
        return (values.clamp_min(floor).log() * preferences).sum(-1)
    return (values * preferences).sum(-1).clamp_min(floor).log()
