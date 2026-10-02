import math
from types import SimpleNamespace

import numpy as np

from rxnflow.config import SubsamplingConfig
from rxnflow.gflownet.policy import SubsamplingPolicy


def test_full_library_has_no_random_draw() -> None:
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    policy = SubsamplingPolicy(
        SimpleNamespace(blocks={"small": range(5)}),
        SubsamplingConfig(min_sampling=10),
        rng,
    )
    indices, weight = policy.sample("small")
    assert indices.tolist() == list(range(5))
    assert weight == 0
    assert before == rng.bit_generator.state
    assert policy.sample("small")[0] is indices


def test_independent_uniform_draw_and_reference_weights() -> None:
    env = SimpleNamespace(blocks={"a": range(19)})
    config = SubsamplingConfig(sampling_ratio=0.25, min_sampling=2)
    first, weight = SubsamplingPolicy(env, config, np.random.default_rng(9)).sample("a")
    second, other_weight = SubsamplingPolicy(
        env, config, np.random.default_rng(9)
    ).sample("a")
    np.testing.assert_array_equal(first, second)
    assert len(first) == 4  # Reference floor, not ceil or observed-count quota.
    assert np.all(first[:-1] < first[1:])
    assert weight == math.log(19 / 4)
    assert weight == other_weight
