"""Uniform library draws independent of the observed training actions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.config import SubsamplingConfig


@dataclass
class BlockSubsample:
    indices: Tensor
    log_importance: float


class BlockSubsampler:
    def __init__(self, size: int, config: SubsamplingConfig):
        self.size = size
        # RxnFlow/CGFlow use floor(N * ratio), with the configured minimum.
        self.count = min(
            size, max(config.min_sampling, int(size * config.sampling_ratio))
        )
        self.log_importance = math.log(size / self.count)
        self.full_indices = torch.arange(size) if self.count == size else None

    def sample(self, generator: torch.Generator) -> BlockSubsample:
        if self.full_indices is not None:
            return BlockSubsample(self.full_indices, 0.0)
        # O(sample size), preserving the agreed removal of full-library shuffles.
        rng = random.Random(int(torch.randint(2**63 - 1, (), generator=generator)))
        indices = torch.tensor(rng.sample(range(self.size), self.count), dtype=torch.long)
        return BlockSubsample(indices, self.log_importance)
