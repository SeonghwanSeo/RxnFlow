"""Subsampled categorical helpers used by training and generation."""

from __future__ import annotations

import torch
from torch import Tensor


def corrected_log_probability(
    selected_logit: Tensor, sampled_logits: Tensor, log_importance: Tensor
) -> Tensor:
    assert sampled_logits.numel() > 0
    estimate = torch.logsumexp(
        sampled_logits + log_importance.to(sampled_logits.device), dim=0
    )
    return selected_logit - estimate


def sample_position(
    logits: Tensor,
    generator: torch.Generator,
    temperature: float = 1.0,
    random_action_prob: float = 0.0,
) -> int:
    assert logits.ndim == 1 and logits.numel() > 0
    if torch.rand((), generator=generator).item() < random_action_prob:
        return int(torch.randint(len(logits), (), generator=generator).item())
    probabilities = torch.softmax(logits.detach().cpu() / temperature, dim=0)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
