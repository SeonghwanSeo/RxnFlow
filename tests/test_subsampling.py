import torch

from rxnflow.config import SubsamplingConfig
from rxnflow.gflownet.categorical import corrected_log_probability
from rxnflow.gflownet.subsampling import UniformActionSpace


def test_ratio_one_preserves_global_indices() -> None:
    action_space = UniformActionSpace(
        5, SubsamplingConfig(sampling_ratio=1.0, min_sampling=50)
    )
    sample = action_space.sample(torch.Generator().manual_seed(1))
    assert sample.indices.tolist() == [0, 1, 2, 3, 4]
    assert torch.all(sample.inclusion_probability == 1)
    assert torch.all(sample.log_importance == 0)


def test_uniform_reproducibility_and_importance() -> None:
    config = SubsamplingConfig(sampling_ratio=0.25, min_sampling=2, importance_temp=0.5)
    action_space = UniformActionSpace(16, config)
    first = action_space.sample(torch.Generator().manual_seed(9))
    second = action_space.sample(torch.Generator().manual_seed(9))
    assert torch.equal(first.indices, second.indices)
    assert len(first.indices) == 4
    assert torch.allclose(
        first.log_importance, first.inclusion_probability.reciprocal().log()
    )
    assert first.indices.unique().numel() == first.indices.numel()


def test_observed_block_is_included_with_conditional_weights() -> None:
    space = UniformActionSpace(8, SubsamplingConfig(sampling_ratio=0.25, min_sampling=1))
    sample = space.sample(torch.Generator().manual_seed(0), required_index=3)
    assert 3 in sample.indices.tolist()
    assert sample.inclusion_probability[sample.indices == 3].item() == 1
    assert torch.allclose(
        sample.inclusion_probability[sample.indices != 3], torch.tensor([1 / 7])
    )
    selected = torch.tensor(1.0, requires_grad=True)
    sampled = torch.stack([selected, selected * 0])
    probability = corrected_log_probability(
        selected, sampled, torch.log(torch.tensor([1.0, 7.0]))
    )
    probability.backward()
    assert selected.grad is not None and probability <= 0
