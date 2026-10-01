import math
from types import SimpleNamespace

from rxnflow.envs.retrosynthesis import (
    MultiRetroSyntheticAnalyzer,
    RetroSynthesisTree,
    RetroSyntheticAnalyzer,
)
from rxnflow.types import ActionKind, RxnAction


class StaticAnalyzer:
    def __init__(self, tree: RetroSynthesisTree):
        self.tree = tree

    def run(self, smiles: str, max_reactions: int) -> RetroSynthesisTree:
        return self.tree


class EchoAnalyzer:
    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ) -> RetroSynthesisTree:
        return RetroSynthesisTree(smiles, known_branches)


class UnexpectedUnaryReaction:
    def run_reverse(self, mol, limit):
        raise AssertionError("deeper reverse reaction was not pruned")


def test_depth_weighted_backward_probability() -> None:
    selected = RxnAction(ActionKind.UNI_REACTION, "CC", reaction="selected")
    alternative = RxnAction(ActionKind.UNI_REACTION, "CC", reaction="alternative")
    leaf = RetroSynthesisTree("")
    one_step = RetroSynthesisTree("one", [(selected, leaf)])
    two_step = RetroSynthesisTree("two", [(alternative, one_step)])
    root = RetroSynthesisTree(
        "root",
        [(selected, one_step), (alternative, two_step)],
    )
    analyzer = MultiRetroSyntheticAnalyzer(StaticAnalyzer(root), workers=0)
    value = analyzer.log_probability(
        "root", 2, selected, total_actions=10, parent_smiles="one"
    )
    assert value is not None
    assert math.isclose(value, math.log(0.1) - math.log(0.1 + 0.01))


def test_known_branch_is_preserved_and_shorter_leaf_prunes_dfs() -> None:
    brick = "[1*]C"
    env = SimpleNamespace(
        uni_reactions={"unexpected": UnexpectedUnaryReaction()},
        bi_actions={},
        blocks={"1": SimpleNamespace(smiles=[brick])},
        brick_types=["1"],
    )
    analyzer = RetroSyntheticAnalyzer(env)
    tree = analyzer.run(brick, max_reactions=2)
    assert tree is not None
    assert [action.kind for action, _ in tree.branches] == [ActionKind.FIRST_BLOCK]

    generated = RxnAction(ActionKind.UNI_REACTION, "CC", reaction="generated")
    child = RetroSynthesisTree(brick, tree.branches)
    empty_env = SimpleNamespace(
        uni_reactions={}, bi_actions={}, blocks={}, brick_types=[]
    )
    known_tree = RetroSyntheticAnalyzer(empty_env).run(
        "CC", max_reactions=2, known_branches=[(generated, child)]
    )
    assert known_tree is not None
    assert known_tree.branches == [(generated, child)]


def test_worker_queue_collects_multiple_submissions() -> None:
    analyzer = MultiRetroSyntheticAnalyzer(EchoAnalyzer(), workers=2)
    action = RxnAction(ActionKind.FIRST_BLOCK, "[1*]C", block_type="1", block_index=0)
    child = RetroSynthesisTree("")
    try:
        analyzer.submit(3, "first", 0, [(action, child)])
        analyzer.submit(7, "second", 0, [(action, child)])
        assert len(analyzer.futures) == 2
        results = analyzer.result()
    finally:
        analyzer.close()
    assert [key for key, _ in results] == [3, 7]
    assert [tree.smiles for _, tree in results if tree is not None] == [
        "first",
        "second",
    ]


def test_same_action_from_different_parents_has_distinct_backward_probability() -> None:
    action = RxnAction(ActionKind.UNI_REACTION, "CC", reaction="conversion")
    leaf = RetroSynthesisTree("")
    first = RetroSynthesisTree("first", [(action, leaf)])
    second = RetroSynthesisTree("second", [(action, leaf)])
    tree = RetroSynthesisTree("CC", [(action, first), (action, second)])
    for parent in ("first", "second"):
        value = MultiRetroSyntheticAnalyzer.tree_log_probability(tree, action, 10, parent)
        assert math.isclose(value, -math.log(2))
