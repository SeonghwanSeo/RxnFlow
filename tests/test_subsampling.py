import torch

from rxnflow.config import SubsamplingConfig
from rxnflow.envs.building_block import BlockLibrary
from rxnflow.policy import TieredActionSpace, block_penalty, corrected_log_probability


def test_ratio_one_preserves_non_contiguous_global_indices() -> None:
    tiers = torch.tensor([3, 1, 3, 5, 1])
    action_space = TieredActionSpace(
        tiers, SubsamplingConfig(sampling_ratio=1.0, min_sampling=50)
    )
    sample = action_space.sample(torch.Generator().manual_seed(1))
    assert sample.indices.tolist() == [0, 1, 2, 3, 4]
    assert torch.all(sample.inclusion_probability == 1)
    assert torch.all(sample.log_importance == 0)


def test_tier_allocation_reproducibility_and_importance() -> None:
    tiers = torch.tensor([1] * 10 + [3] * 4 + [7] * 2)
    config = SubsamplingConfig(sampling_ratio=0.25, min_sampling=2, importance_temp=0.5)
    action_space = TieredActionSpace(tiers, config)
    first = action_space.sample(torch.Generator().manual_seed(9))
    second = action_space.sample(torch.Generator().manual_seed(9))
    assert torch.equal(first.indices, second.indices)
    assert len(first.indices) == sum(action_space.allocation().values())
    assert torch.allclose(
        first.log_importance, first.inclusion_probability.reciprocal().log()
    )
    assert first.indices.unique().numel() == first.indices.numel()


def test_capacity_penalty_and_corrected_probability() -> None:
    library = BlockLibrary(
        block_type="x",
        smiles=["C", "CC", "CCC"],
        identifiers=["1", "2", "3"],
        tiers=torch.tensor([1, 1, 2]),
        properties=torch.zeros((3, 9)),
        fingerprints=torch.zeros((3, 678)),
        heavy_atoms=torch.tensor([1, 2, 3]),
    )
    penalty = block_penalty(
        library,
        torch.tensor([2, 0]),
        current_heavy_atoms=49,
        max_heavy_atoms=50,
    )
    assert penalty.tolist() == [-torch.inf, 0.0]
    selected = torch.tensor(1.0, requires_grad=True)
    probability = corrected_log_probability(
        selected, torch.tensor([0.0, 0.5]), torch.log(torch.tensor([2.0, 2.0]))
    )
    probability.backward()
    assert selected.grad is not None
    assert probability <= 0
