import math
from collections import Counter

import numpy as np
import torch

from rxnflow.core.types import ActionSubspace, ActionType
from rxnflow.gflownet.policy import ActionCategorical, ActionLogits


def test_device_sampling_balances_libraries_and_keeps_property_masks():
    torch.manual_seed(11)
    n = 6000
    # Each library pair and unary action gets mass 1/3 before masks, even
    # when the two libraries share a reaction and have very different sizes.
    small = ActionLogits(
        ActionSubspace(("couple", "small"), ActionType.BIRXN_BRICK, 1),
        torch.full((n, 1), 100.0),
        torch.full((1,), 5.0),
    )
    large = ActionLogits(
        ActionSubspace(("couple", "large"), ActionType.BIRXN_BRICK, 9),
        torch.full((n, 9), 100.0),
        torch.full((9,), 5.0),
    )
    unary = ActionLogits(
        ActionSubspace(("convert", None), ActionType.UNIRXN_TRANSFORM, 1),
        torch.full((n, 1), -100.0),
        torch.zeros(1),
    )
    policy = ActionCategorical(
        [small, large, unary], torch.zeros(n, 2), logit_scale=torch.ones(n, 1)
    )
    counts = Counter(action.library_name for action in policy.sample(1.0, 1.0, 1.0))
    for name in ("small", "large", None):
        assert abs(counts[name] / n - 1 / 3) < 0.03
    # Masking eight of nine large-library columns leaves mass 1/9 there.
    large.logits[:, 1:] = -torch.inf
    counts = Counter(action.library_name for action in policy.sample(1.0, 1.0, 1.0))
    assert abs(counts["large"] / n - 1 / 19) < 0.02
    small.logits.fill_(-torch.inf)
    large.logits.fill_(-torch.inf)
    unary.logits[0] = -torch.inf
    actions = policy.sample(1.0, 1.0, 1.0)
    assert actions[0] is None
    assert all(action.action_type.is_unirxn for action in actions[1:])


def test_policy_sampling_matches_temperature_and_importance_weights():
    torch.manual_seed(23)
    n = 6000
    logits = torch.tensor([0.0, 0.5, -torch.inf])
    weights = torch.tensor([math.log(4), 0.0, 0.0])
    group = ActionLogits(
        ActionSubspace(("couple", "a"), ActionType.BIRXN_BRICK, 3, np.arange(3)),
        logits.repeat(n, 1),
        weights,
    )
    policy = ActionCategorical([group], torch.zeros(n, 2), logit_scale=torch.ones(n, 1))
    actions = policy.sample(0.7, 0.0, 0.5)
    counts = Counter(action.synthon_index for action in actions)
    expected = ((logits + 0.5 * weights) / 0.7).softmax(0)
    assert counts[2] == 0
    assert abs(counts[0] / n - expected[0]) < 0.03
    torch.testing.assert_close(
        policy.log_partition(), torch.logsumexp(logits + weights, 0).expand(n)
    )


def test_subspace_decodes_sampled_library_indices():
    from rxnflow.core.types import Action

    subspace = ActionSubspace(
        ("couple", "a"),
        ActionType.BIRXN_BRICK,
        10,
        np.array([9, 4]),
    )
    assert subspace.action_at(0) == Action(ActionType.BIRXN_BRICK, "couple", "a", 9)
    assert subspace.action_at(1) == Action(ActionType.BIRXN_BRICK, "couple", "a", 4)


def test_full_subspace_and_sampled_copy_decode_without_mutation():
    from dataclasses import replace

    from rxnflow.core.types import Action

    full = ActionSubspace(("first_synthon", "a"), ActionType.FIRST_SYNTHON, 5)
    sampled = replace(full, sample_indices=np.array([4, 1]))
    assert full.sample_indices is None
    assert full.action_at(0) == Action(
        ActionType.FIRST_SYNTHON, library_name="a", synthon_index=0
    )
    assert full.action_at(4) == Action(
        ActionType.FIRST_SYNTHON, library_name="a", synthon_index=4
    )
    assert sampled.action_at(0) == Action(
        ActionType.FIRST_SYNTHON, library_name="a", synthon_index=4
    )
    assert sampled.action_at(1) == Action(
        ActionType.FIRST_SYNTHON, library_name="a", synthon_index=1
    )
    unary = ActionSubspace(("convert", None), ActionType.UNIRXN_TRANSFORM, 1)
    assert unary.action_at(0) == Action(ActionType.UNIRXN_TRANSFORM, "convert")
