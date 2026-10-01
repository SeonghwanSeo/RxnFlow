"""Uniform building-block action-space subsampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.config import SubsamplingConfig


@dataclass(frozen=True)
class BlockSample:
    indices: Tensor
    inclusion_probability: Tensor
    log_importance: Tensor


class UniformActionSpace:
    def __init__(self, size: int, config: SubsamplingConfig):
        assert size > 0
        self.size = size
        self.config = config

    def sample(
        self, generator: torch.Generator, required_index: int | None = None
    ) -> BlockSample:
        count = min(
            self.size,
            max(
                self.config.min_sampling,
                math.ceil(self.size * self.config.sampling_ratio),
            ),
        )
        if required_index is not None:
            assert 0 <= required_index < self.size
            # During TB/replay, include the observed block with probability one
            # and uniformly sample the remaining population. At least one other
            # block is needed to estimate that population when it is nonempty.
            count = min(self.size, max(2, count))
        if count == self.size:
            indices = torch.arange(self.size, dtype=torch.long)
            inclusion = torch.ones(count, dtype=torch.float32)
        elif required_index is not None:
            others = torch.randperm(self.size - 1, generator=generator)[: count - 1]
            others += (others >= required_index).long()
            indices = torch.sort(
                torch.cat([others, torch.tensor([required_index])])
            ).values
            inclusion = torch.full(
                (count,), (count - 1) / (self.size - 1), dtype=torch.float32
            )
            inclusion[indices == required_index] = 1.0
        else:
            indices = torch.randperm(self.size, generator=generator)[:count]
            indices = torch.sort(indices).values
            inclusion = torch.full((count,), count / self.size, dtype=torch.float32)
        return BlockSample(
            indices=indices,
            inclusion_probability=inclusion,
            log_importance=torch.log(inclusion.reciprocal()),
        )
