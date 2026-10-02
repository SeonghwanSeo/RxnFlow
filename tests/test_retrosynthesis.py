import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem

from rxnflow.config import Config, ModelConfig
from rxnflow.core.reaction import BiReaction, UniReaction
from rxnflow.core.types import Action, ActionType
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.retrosynthesis import (
    RetroSynthesisAnalyzer,
    Worker,
)
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel


class UnexpectedBinaryReaction:
    def run_reverse(self, mol):
        raise AssertionError("extra synthon decomposition was not pruned")


@pytest.fixture
def policy(prepared_env):
    env = SynthesisEnv(prepared_env)
    config = Config(model=ModelConfig(num_emb=8, num_layers=1, num_synthon_emb=8))
    model = RxnFlowModel(env, config.model, num_objectives=1)
    return RxnFlowPolicy(
        env, model, config, torch.device("cpu"), np.random.default_rng(0)
    )


@pytest.mark.parametrize("penalty", [100.0, 1000.0])
def test_synthon_weighted_backward_probability(policy, penalty) -> None:
    policy.config.training.backward_synthon_penalty = penalty
    selected = Action(ActionType.UNIRXN_TRANSFORM, reaction="selected")
    alternative = Action(ActionType.UNIRXN_TRANSFORM, reaction="alternative")
    first = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    coupling = Action(
        ActionType.BIRXN_LINKER, reaction="join", library_name="1-1", synthon_index=0
    )
    routes = [
        [(selected, "one"), (first, "")],
        [(selected, "one"), (coupling, "two"), (first, "")],
        [(alternative, "two"), (selected, "one"), (first, "")],
    ]
    value = policy.calc_bck_logprob(selected, routes, "one")
    expected = math.log(1 + 1 / penalty) - math.log(2 + 1 / penalty)
    assert math.isclose(value, expected)
    assert policy.calc_bck_logprob(first, [[(first, "")]], "") == 0.0
    assert policy.calc_bck_logprob(selected, [], "one") is None
    assert policy.calc_bck_logprob(selected, routes, "missing") is None


def test_known_routes_are_preserved_and_reaction_budget_bounds_dfs() -> None:
    brick = "[1*]C"
    env = SimpleNamespace(
        uni_reactions={},
        bi_reactions={"unexpected": UnexpectedBinaryReaction()},
        synthons={"1": SimpleNamespace(smiles=["*C"])},
        brick_types=["1"],
    )
    analyzer = Worker(env)
    routes = analyzer.run(brick, max_reactions=0)
    first = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    assert routes == [[(first, "")]]
    # A catalog origin prunes extra synthons, independently of reaction depth.
    assert analyzer.run(brick, max_reactions=3) == routes
    # Seeding an already discoverable branch must not duplicate its mass.
    assert analyzer.run(brick, max_reactions=0, known_trajectories=routes) == routes

    generated = Action(ActionType.UNIRXN_TRANSFORM, reaction="generated")
    known = [[(generated, brick), *route] for route in routes]
    empty_env = SimpleNamespace(
        uni_reactions={}, bi_reactions={}, synthons={}, brick_types=[]
    )
    worker = Worker(empty_env)
    assert worker.run("CC", max_reactions=2, known_trajectories=known) == known
    assert worker.run("CC", max_reactions=2) == []


def test_reverse_search_finds_catalog_match_after_second_decomposition() -> None:
    # Five distinct cuts of hexane; only the third has both fragments in this
    # catalog. Truncating canonical products to two silently loses this route.
    reaction = BiReaction(
        "join",
        "[#6:1]-[1*].[#6:2]-[2*]>>[#6:1]-[#6:2]",
        "[#6:1]-[#6:2]>>[#6:1]-[1*].[#6:2]-[2*]",
        (1, 2),
    )
    product = Chem.MolFromSmiles("CCCCCC")
    assert len(reaction.run_reverse(product)) == 5
    env = SimpleNamespace(
        uni_reactions={},
        bi_reactions={"join": reaction},
        synthons={name: SimpleNamespace(smiles=["*CCC"]) for name in ("1", "2")},
        brick_types=["1", "2"],
    )
    routes = Worker(env).run("CCCCCC", max_reactions=1)
    assert len(routes) == 1
    action, parent = routes[0][0]
    assert action.library_name == "2" and parent == "[1*]CCC"
    assert routes[0][-1] == (
        Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0),
        "",
    )


def test_same_synthon_count_allows_more_unary_steps_and_preserves_generated_route() -> (
    None
):
    close = UniReaction("close", "[#6:1]-[1*]>>[#6:1]", "[#6:1]>>[#6:1]-[1*]", 1, None)
    activate = UniReaction(
        "activate",
        "[#6:1]-[33*]>>[#6:1]-[1*]",
        "[#6:1]-[1*]>>[#6:1]-[33*]",
        33,
        1,
    )
    env = SimpleNamespace(
        uni_reactions={"close": close, "activate": activate},
        bi_reactions={},
        synthons={name: SimpleNamespace(smiles=["*CC"]) for name in ("1", "33")},
        brick_types=["1", "33"],
    )
    analyzer = Worker(env)
    assert [len(route) for route in analyzer.run("CC", max_reactions=1)] == [2]
    routes = analyzer.run("CC", max_reactions=2)
    assert sorted(map(len, routes)) == [2, 3]
    assert all(route[-1][0].action_type == ActionType.FIRST_SYNTHON for route in routes)
    assert all(route[-1][1] == "" for route in routes)
    # The longer generated history remains represented in the backward mass.
    known = [
        [
            (Action(ActionType.UNIRXN_TERMINAL, reaction="close"), "[1*]CC"),
            (Action(ActionType.UNIRXN_TRANSFORM, reaction="activate"), "[33*]CC"),
            (Action(ActionType.FIRST_SYNTHON, library_name="33", synthon_index=0), ""),
        ]
    ]
    assert analyzer.run("CC", max_reactions=2, known_trajectories=known) == known
    # Root branches are preserved as supplied. Seed at the precursor to allow
    # discovery of a shorter FirstSynthon route alongside the generated history.
    precursor_routes = analyzer.run(
        "[1*]CC", max_reactions=1, known_trajectories=[known[0][1:]]
    )
    assert sorted(map(len, precursor_routes)) == [1, 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_worker_queue_collects_multiple_submissions(workers) -> None:
    env = SimpleNamespace(uni_reactions={}, bi_reactions={}, synthons={}, brick_types=[])
    analyzer = RetroSynthesisAnalyzer(env, workers=workers)
    action = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    known = [[(action, "")]]
    try:
        analyzer.submit(3, "CC", 0, known)
        analyzer.submit(7, "CCC", 0, known)
        results = analyzer.result()
        assert analyzer.result() == []
    finally:
        analyzer.close()
    assert results == [(3, known), (7, known)]


def test_same_action_from_different_parents_has_distinct_backward_probability(
    policy,
) -> None:
    action = Action(ActionType.UNIRXN_TRANSFORM, reaction="conversion")
    first = Action(ActionType.FIRST_SYNTHON, library_name="1", synthon_index=0)
    routes = [
        [(action, "first"), (first, "")],
        [(action, "second"), (action, "third"), (first, "")],
    ]
    for parent in ("first", "second"):
        assert math.isclose(policy.calc_bck_logprob(action, routes, parent), -math.log(2))
