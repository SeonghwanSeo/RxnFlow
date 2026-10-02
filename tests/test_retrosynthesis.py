import math
from types import SimpleNamespace

from rdkit import Chem

from rxnflow.core.reaction import BiReaction, UniReaction
from rxnflow.core.types import Action, ActionType
from rxnflow.envs.retrosynthesis import (
    RetroSynthesisAnalyzer,
    RetrosynthesisTree,
    Worker,
)


class UnexpectedUnaryReaction:
    def run_reverse(self, mol):
        raise AssertionError("deeper reverse reaction was not pruned")


def test_depth_weighted_backward_probability() -> None:
    selected = Action(ActionType.UNI_REACTION, reaction="selected")
    alternative = Action(ActionType.UNI_REACTION, reaction="alternative")
    leaf = RetrosynthesisTree("")
    one_step = RetrosynthesisTree("one", [(selected, leaf)])
    two_step = RetrosynthesisTree("two", [(alternative, one_step)])
    root = RetrosynthesisTree(
        "root",
        [(selected, one_step), (alternative, two_step)],
    )
    value = RetroSynthesisAnalyzer.tree_log_probability(
        root, selected, total_actions=10, parent_smiles="one"
    )
    assert value is not None
    assert math.isclose(value, math.log(0.1) - math.log(0.1 + 0.01))


def test_known_branch_is_preserved_and_reaction_budget_bounds_dfs() -> None:
    brick = "[1*]C"
    env = SimpleNamespace(
        uni_reactions={"unexpected": UnexpectedUnaryReaction()},
        bi_reactions={},
        blocks={"1": SimpleNamespace(smiles=["*C"])},
        brick_types=["1"],
    )
    analyzer = Worker(env)
    tree = analyzer.run(brick, max_reactions=0)
    assert tree is not None
    assert [action.action_type for action, _ in tree.branches] == [ActionType.FIRST_BLOCK]

    generated = Action(ActionType.UNI_REACTION, reaction="generated")
    child = RetrosynthesisTree(brick, tree.branches)
    empty_env = SimpleNamespace(
        uni_reactions={}, bi_reactions={}, blocks={}, brick_types=[]
    )
    known_tree = Worker(empty_env).run(
        "CC", max_reactions=2, known_branches=[(generated, child)]
    )
    assert known_tree is not None
    assert known_tree.branches == [(generated, child)]


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
        blocks={name: SimpleNamespace(smiles=["*CCC"]) for name in ("1", "2")},
        brick_types=["1", "2"],
    )
    tree = Worker(env).run("CCCCCC", max_reactions=1)
    assert tree is not None
    assert len(tree.branches) == 1
    action, child = tree.branches[0]
    assert action.block_type == "2" and child.smiles == "[1*]CCC"
    assert child.branches[0][0].action_type == ActionType.FIRST_BLOCK


def test_short_route_does_not_hide_longer_route_within_budget() -> None:
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
        blocks={name: SimpleNamespace(smiles=["*CC"]) for name in ("1", "33")},
        brick_types=["1", "33"],
    )
    analyzer = Worker(env)
    assert analyzer.run("CC", max_reactions=1).leaf_depths() == [2]
    assert sorted(analyzer.run("CC", max_reactions=2).leaf_depths()) == [2, 3]


def test_worker_queue_collects_multiple_submissions() -> None:
    env = SimpleNamespace(
        uni_reactions={}, bi_reactions={}, blocks={}, brick_types=[]
    )
    analyzer = RetroSynthesisAnalyzer(env, workers=2)
    action = Action(ActionType.FIRST_BLOCK, block_type="1", block_index=0)
    child = RetrosynthesisTree("")
    try:
        analyzer.submit(3, "CC", 0, [(action, child)])
        analyzer.submit(7, "CCC", 0, [(action, child)])
        assert len(analyzer.futures) == 2
        results = analyzer.result()
    finally:
        analyzer.close()
    assert [key for key, _ in results] == [3, 7]
    assert [tree.smiles for _, tree in results if tree is not None] == [
        "CC",
        "CCC",
    ]


def test_same_action_from_different_parents_has_distinct_backward_probability() -> None:
    action = Action(ActionType.UNI_REACTION, reaction="conversion")
    leaf = RetrosynthesisTree("")
    first = RetrosynthesisTree("first", [(action, leaf)])
    second = RetrosynthesisTree("second", [(action, leaf)])
    tree = RetrosynthesisTree("CC", [(action, first), (action, second)])
    for parent in ("first", "second"):
        value = RetroSynthesisAnalyzer.tree_log_probability(tree, action, 10, parent)
        assert math.isclose(value, -math.log(2))
