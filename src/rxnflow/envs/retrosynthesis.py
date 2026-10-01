"""Retrosynthetic route enumeration for synthon trajectories."""

from __future__ import annotations

import math
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field

from rdkit import Chem

from rxnflow.envs.chemistry.synthon import typed_dummy_isotopes
from rxnflow.types import ActionKind, RxnAction


@dataclass
class RetroSynthesisTree:
    smiles: str
    branches: list[tuple[RxnAction, RetroSynthesisTree]] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.branches

    def leaf_depths(self, depth: int = 0) -> list[int]:
        if self.is_leaf:
            return [depth]
        return [
            value for _, child in self.branches for value in child.leaf_depths(depth + 1)
        ]


class RetroSyntheticAnalyzer:
    def __init__(self, env):
        self.uni_reactions = env.uni_reactions
        self.bi_actions = env.bi_actions
        self.block_search = {
            block_type: {smiles: index for index, smiles in enumerate(library.smiles)}
            for block_type, library in env.blocks.items()
        }
        self.brick_types = set(env.brick_types)
        self.compatible = {
            name: [
                block_type
                for block_type, library in env.blocks.items()
                if action.block_site_type in library.site_types
            ]
            for name, action in self.bi_actions.items()
        }
        self._memo: dict[tuple[str, int], RetroSynthesisTree | None] = {}
        self._max_depth = 0
        self._min_depth = 0

    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        self._max_depth = self._min_depth = max_reactions + 1
        if known_branches:
            self._min_depth = min(
                min(child.leaf_depths()) + 1 for _, child in known_branches
            )
        self._memo = {}
        return self._dfs(smiles, 1, known_branches)

    def _check_depth(self, depth: int) -> bool:
        return depth <= self._max_depth and depth <= self._min_depth

    def _dfs(
        self,
        smiles: str,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if not smiles or not self._check_depth(depth):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        key = (canonical, depth)
        if known_branches is None and key in self._memo:
            return self._memo[key]
        branches = list(known_branches or [])
        # The same forward action/outcome may be reachable from different
        # precursors. Backward choices identify both the action and its parent.
        branch_keys = {(action, child.smiles) for action, child in branches}

        if len(typed_dummy_isotopes(mol)) == 1:
            for block_type in sorted(self.brick_types):
                block_index = self.block_search[block_type].get(canonical)
                if block_index is not None:
                    action = RxnAction(
                        ActionKind.FIRST_BLOCK,
                        product_smiles=canonical,
                        block_type=block_type,
                        block_index=block_index,
                    )
                    if (action, "") not in branch_keys:
                        branches.append((action, RetroSynthesisTree("")))
                        branch_keys.add((action, ""))
                    self._min_depth = min(self._min_depth, depth)

        if self._check_depth(depth + 1):
            for name, reaction in self.uni_reactions.items():
                expected = () if reaction.output_type is None else (reaction.output_type,)
                if typed_dummy_isotopes(mol) != expected:
                    continue
                for products in reaction.run_reverse(mol, 2):
                    if len(products) != 1:
                        continue
                    precursor = Chem.MolFromSmiles(products[0])
                    if precursor is None or typed_dummy_isotopes(precursor) != (
                        reaction.input_type,
                    ):
                        continue
                    parent_smiles = Chem.MolToSmiles(precursor)
                    action = RxnAction(ActionKind.UNI_REACTION, canonical, reaction=name)
                    if (action, parent_smiles) in branch_keys:
                        continue
                    if canonical not in reaction.run_forward(precursor):
                        continue
                    child = self._dfs(parent_smiles, depth + 1)
                    if child is not None:
                        branches.append((action, child))
                        branch_keys.add((action, parent_smiles))

            for name, action in self.bi_actions.items():
                for child_smiles, block_smiles in action.reaction.reverse_pairs(
                    mol, action.block_first, 2
                ):
                    child_mol = Chem.MolFromSmiles(child_smiles)
                    block_mol = Chem.MolFromSmiles(block_smiles)
                    if child_mol is None or block_mol is None:
                        continue
                    child_canonical = Chem.MolToSmiles(child_mol)
                    block_canonical = Chem.MolToSmiles(block_mol)
                    if typed_dummy_isotopes(child_mol) != (action.state_type,):
                        continue
                    for block_type in self.compatible[name]:
                        block_index = self.block_search[block_type].get(block_canonical)
                        if block_index is None:
                            continue
                        reverse_action = RxnAction(
                            ActionKind.BI_REACTION,
                            product_smiles=canonical,
                            reaction=name,
                            block_type=block_type,
                            block_index=block_index,
                        )
                        if (reverse_action, child_canonical) in branch_keys:
                            continue
                        if canonical not in action.reaction.run(
                            child_mol, block_mol, action.block_first
                        ):
                            continue
                        child = self._dfs(child_canonical, depth + 1)
                        if child is not None:
                            branches.append((reverse_action, child))
                            branch_keys.add((reverse_action, child_canonical))

        result = RetroSynthesisTree(canonical, branches) if branches else None
        if known_branches is None:
            self._memo[key] = result
        return result


_WORKER_ANALYZER: RetroSyntheticAnalyzer | None = None


def _init_worker(analyzer: RetroSyntheticAnalyzer) -> None:
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = analyzer


def _worker_run(
    smiles: str,
    max_reactions: int,
    known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None,
) -> RetroSynthesisTree | None:
    assert _WORKER_ANALYZER is not None
    return _WORKER_ANALYZER.run(smiles, max_reactions, known_branches)


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer: RetroSyntheticAnalyzer, workers: int):
        self.analyzer = analyzer
        self.pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(analyzer,),
            )
            if workers > 0
            else None
        )
        self.futures: list[tuple[int, Future[RetroSynthesisTree | None]]] = []
        self.results: list[tuple[int, RetroSynthesisTree | None]] = []

    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        if self.pool is None:
            if known_branches is None:
                return self.analyzer.run(smiles, max_reactions)
            return self.analyzer.run(smiles, max_reactions, known_branches)
        return self.pool.submit(
            _worker_run, smiles, max_reactions, known_branches
        ).result()

    def submit(
        self,
        key: int,
        smiles: str,
        max_reactions: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ) -> None:
        if self.pool is None:
            self.results.append(
                (
                    key,
                    self.analyzer.run(smiles, max_reactions, known_branches),
                )
            )
        else:
            future = self.pool.submit(_worker_run, smiles, max_reactions, known_branches)
            self.futures.append((key, future))

    def result(self) -> list[tuple[int, RetroSynthesisTree | None]]:
        if self.pool is None:
            results = self.results
            self.results = []
            return results
        results = [(key, future.result()) for key, future in self.futures]
        self.futures = []
        return results

    @staticmethod
    def tree_log_probability(
        tree: RetroSynthesisTree | None,
        action: RxnAction,
        total_actions: int,
        parent_smiles: str,
    ) -> float | None:
        if tree is None:
            return None
        numerator = 0.0
        denominator = 0.0
        for branch_action, child in tree.branches:
            weight = sum(total_actions ** (-depth) for depth in child.leaf_depths())
            denominator += weight
            if branch_action == action and child.smiles == parent_smiles:
                numerator += weight
        if numerator <= 0 or denominator <= 0:
            return None
        return math.log(numerator) - math.log(denominator)

    def log_probability(
        self,
        smiles: str,
        max_reactions: int,
        action: RxnAction,
        total_actions: int,
        parent_smiles: str,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> float | None:
        tree = self.run(smiles, max_reactions, known_branches)
        return self.tree_log_probability(tree, action, total_actions, parent_smiles)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
