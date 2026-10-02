"""Retrosynthetic route enumeration for synthon trajectories."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

from rdkit import Chem

from rxnflow.core.synthon import typed_dummy_isotopes
from rxnflow.core.types import Action, ActionType, BackwardTrajectory

if TYPE_CHECKING:
    from rxnflow.envs.env import SynthesisEnv


class Worker:
    """Enumerate catalog-supported routes within the supplied reaction budget.

    Depth bounds the search, including cycles; finding a shorter route does
    not prune other branches. Candidates must match a catalog entry and reproduce
    the product in the forward direction. Return reverse-ordered edge lists;
    probability weighting belongs to the policy that consumes these routes.
    """

    def __init__(self, env: SynthesisEnv):
        self.uni_reactions = env.uni_reactions
        self.bi_reactions = env.bi_reactions
        self.block_search = {
            block_type: {smiles: index for index, smiles in enumerate(library.smiles)}
            for block_type, library in env.blocks.items()
        }
        self.brick_types = set(env.brick_types)
        self._memo: dict[tuple[str, int], list[BackwardTrajectory]] = {}
        self._max_depth = 0

    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        self._max_depth = max_reactions + 1  # Include FirstBlock.
        self._memo = {}
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return []
        return self._dfs(mol, Chem.MolToSmiles(mol), 1, known_trajectories)

    def _dfs(
        self,
        mol: Chem.Mol,
        canonical: str,
        depth: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        # 1. Reuse this search's depth-specific results and preserve the known route.
        if depth > self._max_depth:
            return []
        key = (canonical, depth)
        if known_trajectories is None and key in self._memo:
            return self._memo[key]
        trajectories = list(known_trajectories or [])
        # First edges identify backward choices. Retain known suffixes and skip
        # rediscovering their action/parent pair during this root search.
        branch_keys = {trajectory[0] for trajectory in trajectories}

        # 2. Look for a direct FirstBlock origin by restoring the catalog marker.
        signature = typed_dummy_isotopes(mol)
        if len(signature) == 1:
            site_type = signature[0]
            brick = Chem.Mol(mol)
            for atom in brick.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            brick_smiles = Chem.MolToSmiles(brick)
            block_type = str(site_type)
            if block_type in self.brick_types:
                block_index = self.block_search[block_type].get(brick_smiles)
                if block_index is not None:
                    action = Action(
                        ActionType.FIRST_BLOCK,
                        block_type=block_type,
                        block_index=block_index,
                    )
                    if (action, "") not in branch_keys:
                        trajectories.append([(action, "")])
                        branch_keys.add((action, ""))

        # 3. Reverse unary transformations and verify each precursor forward.
        if depth < self._max_depth:
            for name, reaction in self.uni_reactions.items():
                expected = () if reaction.output_type is None else (reaction.output_type,)
                if signature != expected:
                    continue
                for products in reaction.run_reverse(mol):
                    if len(products) != 1:
                        continue
                    precursor = products[0]
                    if typed_dummy_isotopes(precursor) != (reaction.input_type,):
                        continue
                    parent_smiles = Chem.MolToSmiles(precursor)
                    action = Action(ActionType.UNI_REACTION, reaction=name)
                    if (action, parent_smiles) in branch_keys:
                        continue
                    forward_product = reaction.run_forward(precursor)
                    if (
                        forward_product is None
                        or Chem.MolToSmiles(forward_product) != canonical
                    ):
                        continue
                    suffixes = self._dfs(precursor, parent_smiles, depth + 1)
                    if suffixes:
                        trajectories.extend(
                            [(action, parent_smiles), *suffix] for suffix in suffixes
                        )
                        branch_keys.add((action, parent_smiles))

            # 4. Reverse couplings, recover the oriented block, and find its row.
            for name, action in self.bi_reactions.items():
                for child_mol, block_mol in action.run_reverse(mol):
                    child_canonical = Chem.MolToSmiles(child_mol)
                    block_canonical = Chem.MolToSmiles(block_mol)
                    if typed_dummy_isotopes(child_mol) != (action.state_type,):
                        continue
                    # Reverse products carry the incoming isotope-0 attachment
                    # and (for linkers) the remaining type. Together with the
                    # reaction's incoming type this determines one library.
                    block_sites = typed_dummy_isotopes(block_mol)
                    if not block_sites or block_sites[0] != 0:
                        continue
                    block_type = "-".join(map(str, (action.block_type, *block_sites[1:])))
                    library = self.block_search.get(block_type)
                    if library is None:
                        continue
                    block_index = library.get(block_canonical)
                    if block_index is None:
                        continue
                    reverse_action = Action(
                        ActionType.BI_REACTION,
                        reaction=name,
                        block_type=block_type,
                        block_index=block_index,
                    )
                    if (reverse_action, child_canonical) in branch_keys:
                        continue
                    forward_product = action.run_forward(child_mol, block_mol)
                    if (
                        forward_product is None
                        or Chem.MolToSmiles(forward_product) != canonical
                    ):
                        continue
                    suffixes = self._dfs(child_mol, child_canonical, depth + 1)
                    if suffixes:
                        trajectories.extend(
                            [(reverse_action, child_canonical), *suffix]
                            for suffix in suffixes
                        )
                        branch_keys.add((reverse_action, child_canonical))

        # 5. Cache only unseeded searches; known routes are specific to a rollout.
        if known_trajectories is None:
            self._memo[key] = trajectories
        return trajectories


_WORKER: Worker | None = None


def _init_worker(worker: Worker) -> None:
    global _WORKER
    _WORKER = worker


def _worker_run(
    smiles: str,
    max_reactions: int,
    known_trajectories: list[BackwardTrajectory] | None,
) -> list[BackwardTrajectory]:
    assert _WORKER is not None
    return _WORKER.run(smiles, max_reactions, known_trajectories)


class RetroSynthesisAnalyzer:
    """Run reverse searches locally or in workers and return backward trajectories."""

    def __init__(self, env: SynthesisEnv, workers: int):
        self.worker = Worker(env)
        self.pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(self.worker,),
            )
            if workers > 0
            else None
        )
        self.futures: list[tuple[int, Future[list[BackwardTrajectory]]]] = []
        self.results: list[tuple[int, list[BackwardTrajectory]]] = []

    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        if self.pool is None:
            return self.worker.run(smiles, max_reactions, known_trajectories)
        return self.pool.submit(
            _worker_run, smiles, max_reactions, known_trajectories
        ).result()

    def submit(
        self,
        key: int,
        smiles: str,
        max_reactions: int,
        known_trajectories: list[BackwardTrajectory],
    ) -> None:
        if self.pool is None:
            self.results.append(
                (
                    key,
                    self.worker.run(smiles, max_reactions, known_trajectories),
                )
            )
        else:
            future = self.pool.submit(
                _worker_run, smiles, max_reactions, known_trajectories
            )
            self.futures.append((key, future))

    def result(self) -> list[tuple[int, list[BackwardTrajectory]]]:
        """Wait for pending searches and drain their results in submission order."""
        if self.pool is None:
            results = self.results
            self.results = []
            return results
        results = [(key, future.result()) for key, future in self.futures]
        self.futures = []
        return results

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
