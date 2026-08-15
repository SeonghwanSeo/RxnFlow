"""Tier-stratified building-block action-space subsampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.config import SubsamplingConfig


@dataclass(frozen=True)
class TierSample:
    indices: Tensor
    inclusion_probability: Tensor
    log_importance: Tensor


class TieredActionSpace:
    """Sample global block indices without replacement within each price tier."""

    def __init__(self, tiers: Tensor, config: SubsamplingConfig):
        tiers = torch.as_tensor(tiers, dtype=torch.long, device="cpu")
        assert tiers.ndim == 1 and len(tiers) > 0 and (tiers > 0).all()
        self.tiers = tiers
        self.config = config
        self.groups = {
            int(tier): torch.nonzero(tiers == tier, as_tuple=False).flatten()
            for tier in torch.unique(tiers, sorted=True).tolist()
        }

    def allocation(self) -> dict[int, int]:
        sizes = {tier: len(indices) for tier, indices in self.groups.items()}
        if self.config.sampling_ratio == 1:
            return sizes
        requested = min(
            len(self.tiers),
            max(1, math.ceil(len(self.tiers) * self.config.sampling_ratio)),
        )
        allocation = {
            tier: min(size, self.config.min_sampling) for tier, size in sizes.items()
        }
        target = min(len(self.tiers), max(requested, sum(allocation.values())))
        remaining = target - sum(allocation.values())
        tiers = sorted(sizes)
        while remaining:
            for tier in tiers:
                if allocation[tier] < sizes[tier]:
                    allocation[tier] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
        return allocation

    def sample(self, generator: torch.Generator) -> TierSample:
        selected: list[Tensor] = []
        probabilities: list[Tensor] = []
        for tier, count in self.allocation().items():
            global_indices = self.groups[tier]
            size = len(global_indices)
            if count >= size:
                chosen = global_indices.clone()
            else:
                positions = torch.randperm(size, generator=generator)[:count]
                chosen = global_indices[positions]
            selected.append(chosen)
            probabilities.append(
                torch.full((len(chosen),), count / size, dtype=torch.float32)
            )
        indices = torch.cat(selected)
        inclusion = torch.cat(probabilities)
        order = torch.argsort(indices)
        indices = indices[order]
        inclusion = inclusion[order]
        importance = torch.log(inclusion.reciprocal())
        return TierSample(
            indices=indices, inclusion_probability=inclusion, log_importance=importance
        )
