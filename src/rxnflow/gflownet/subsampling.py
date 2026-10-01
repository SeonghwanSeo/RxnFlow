"""Uniform building-block action-space subsampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.config import SubsamplingConfig


@dataclass(frozen=True)
class BlockSubsample:
    indices: Tensor
    inclusion_probability: Tensor
    log_importance: Tensor


class BlockSubsampler:
    def __init__(self, size: int, config: SubsamplingConfig):
        assert size > 0
        self.size = size
        self.config = config

    def sample(
        self,
        generator: torch.Generator,
        required_indices: tuple[int, ...] = (),
    ) -> BlockSubsample:
        # One draw per library and policy batch. TB/replay conditions that draw
        # on the union of observed rows, so all state denominators share it.
        size = self.size
        required = torch.tensor(sorted(set(required_indices)), dtype=torch.long)
        n_required = len(required)
        assert all(0 <= index < size for index in required_indices)
        count = min(
            size,
            max(
                self.config.min_sampling,
                math.ceil(size * self.config.sampling_ratio),
                n_required + 1 if n_required else 0,
            ),
        )
        if count == size:
            indices = torch.arange(size, dtype=torch.long)
            inclusion = torch.ones(count, dtype=torch.float32)
        else:
            # Draw compact ranks in the non-required population, then skip the
            # forced rows. No full-library permutation or rejection loop.
            rng = random.Random(int(torch.randint(2**63 - 1, (), generator=generator)))
            others = torch.tensor(
                rng.sample(range(size - n_required), count - n_required),
                dtype=torch.long,
            )
            others += torch.searchsorted(
                required - torch.arange(n_required), others, right=True
            )
            indices = torch.sort(torch.cat([others, required])).values
            inclusion = torch.full(
                (count,),
                (count - n_required) / (size - n_required),
                dtype=torch.float32,
            )
            inclusion[torch.isin(indices, required)] = 1.0
        return BlockSubsample(
            indices=indices,
            inclusion_probability=inclusion,
            log_importance=torch.log(inclusion.reciprocal()),
        )
