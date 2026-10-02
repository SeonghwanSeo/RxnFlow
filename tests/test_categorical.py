import math
from collections import Counter

import numpy as np
import torch

from rxnflow.core.types import ActionSubspace, ActionType
from rxnflow.gflownet.policy import ActionCategorical, ActionLogits


def test_device_sampling_balances_libraries_and_keeps_property_masks():
    torch.manual_seed(11)
    n = 6000
    # One action group has libraries of unequal size; the other is unary.
    # At temperature 1 the random policy gives each group mass 1/2 and
    # each of the two block libraries mass 1/4, irrespective of library size.
    blocks = ActionLogits(
        ActionSubspace(
            "couple",
            ActionType.BI_REACTION,
            ["small", "large"],
            [1, 9],
        ),
        torch.full((n, 10), 100.0),
        torch.full((10,), 5.0),
    )
    unary = ActionLogits(
        ActionSubspace("convert", ActionType.UNI_REACTION, [], []),
        torch.full((n, 1), -100.0),
        torch.zeros(1),
    )
    policy = ActionCategorical(
        [blocks, unary], torch.zeros(n, 2), logit_scale=torch.ones(n, 1)
    )
    counts = Counter(action.block_type for action in policy.sample(1.0, 1.0, 1.0))
    for name, expected in (("small", 0.25), ("large", 0.25), (None, 0.5)):
        assert abs(counts[name] / n - expected) < 0.03
    # Retained but masked columns must stay impossible even under random policy.
    blocks.logits.fill_(-torch.inf)
    unary.logits[0] = -torch.inf
    actions = policy.sample(1.0, 1.0, 1.0)
    assert actions[0] is None
    assert all(action.action_type == ActionType.UNI_REACTION for action in actions[1:])


def test_policy_sampling_matches_temperature_and_importance_weights():
    torch.manual_seed(23)
    n = 6000
    logits = torch.tensor([0.0, 0.5, -torch.inf])
    weights = torch.tensor([math.log(4), 0.0, 0.0])
    group = ActionLogits(
        ActionSubspace("couple", ActionType.BI_REACTION, ["a"], [3], [np.arange(3)]),
        logits.repeat(n, 1),
        weights,
    )
    policy = ActionCategorical([group], torch.zeros(n, 2), logit_scale=torch.ones(n, 1))
    actions = policy.sample(0.7, 0.0, 0.5)
    counts = Counter(action.block_index for action in actions)
    expected = ((logits + 0.5 * weights) / 0.7).softmax(0)
    assert counts[2] == 0
    assert abs(counts[0] / n - expected[0]) < 0.03
    torch.testing.assert_close(
        policy.log_partition(), torch.logsumexp(logits + weights, 0).expand(n)
    )


def test_subspace_decodes_sampled_library_indices():
    from rxnflow.core.types import Action

    subspace = ActionSubspace(
        "couple",
        ActionType.BI_REACTION,
        ["a", "b"],
        [10, 13],
        [np.array([9, 4]), np.array([12])],
    )
    assert subspace.action_at(0) == Action(ActionType.BI_REACTION, "couple", "a", 9)
    assert subspace.action_at(1) == Action(ActionType.BI_REACTION, "couple", "a", 4)
    assert subspace.action_at(2) == Action(ActionType.BI_REACTION, "couple", "b", 12)


def test_full_subspace_and_sampled_copy_decode_without_mutation():
    from dataclasses import replace

    from rxnflow.core.types import Action

    full = ActionSubspace("first_block", ActionType.FIRST_BLOCK, ["a", "b"], [3, 5])
    sampled = replace(full, sample_indices=[np.array([2]), np.array([4, 1])])
    assert full.sample_indices is None
    assert full.action_at(2) == Action(
        ActionType.FIRST_BLOCK, block_type="a", block_index=2
    )
    assert full.action_at(3) == Action(
        ActionType.FIRST_BLOCK, block_type="b", block_index=0
    )
    assert full.action_at(7) == Action(
        ActionType.FIRST_BLOCK, block_type="b", block_index=4
    )
    assert sampled.action_at(1) == Action(
        ActionType.FIRST_BLOCK, block_type="b", block_index=4
    )
    assert sampled.action_at(2) == Action(
        ActionType.FIRST_BLOCK, block_type="b", block_index=1
    )
    unary = ActionSubspace("convert", ActionType.UNI_REACTION, [], [])
    assert unary.action_at(0) == Action(ActionType.UNI_REACTION, "convert")
