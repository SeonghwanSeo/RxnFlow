import math

import torch

from rxnflow.config import SubsamplingConfig
from rxnflow.gflownet.subsampling import BlockSubsampler


def test_full_library_has_no_random_draw() -> None:
    generator = torch.Generator().manual_seed(1)
    before = generator.get_state()
    sample = BlockSubsampler(5, SubsamplingConfig(min_sampling=10)).sample(generator)
    assert sample.indices.tolist() == list(range(5))
    assert sample.log_importance == 0
    assert torch.equal(before, generator.get_state())


def test_independent_uniform_draw_and_reference_weights() -> None:
    space = BlockSubsampler(19, SubsamplingConfig(sampling_ratio=0.25, min_sampling=2))
    first = space.sample(torch.Generator().manual_seed(9))
    second = space.sample(torch.Generator().manual_seed(9))
    assert torch.equal(first.indices, second.indices)
    assert len(first.indices) == 4  # Reference floor, not ceil or observed-count quota.
    assert first.indices.unique().numel() == 4
    assert first.log_importance == math.log(19 / 4)
    # Budget masks do not change the draw's full-library weight.
    assert first.log_importance == second.log_importance
