import math

import numpy as np
import pytest

from rxnflow.gflownet.policy import SubsamplingPolicy


@pytest.mark.parametrize("num_actions", [1, 5])
def test_full_library_has_no_random_draw(num_actions: int) -> None:
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    policy = SubsamplingPolicy(num_actions, 0.05, 10, rng)
    indices, weight = policy.sample()
    assert indices.tolist() == list(range(num_actions))
    assert weight == 0
    assert before == rng.bit_generator.state
    assert policy.sample()[0] is indices


def test_independent_uniform_draw_and_reference_weights() -> None:
    first, weight = SubsamplingPolicy(19, 0.25, 2, np.random.default_rng(9)).sample()
    second, other_weight = SubsamplingPolicy(
        19, 0.25, 2, np.random.default_rng(9)
    ).sample()
    np.testing.assert_array_equal(first, second)
    assert len(first) == 4  # floor(19 * 0.25); the minimum of 2 is inactive.
    assert np.all(first[:-1] < first[1:])
    assert weight == math.log(19 / 4)
    assert weight == other_weight
