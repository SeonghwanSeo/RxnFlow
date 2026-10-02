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

    The fewest synthons found bounds further exploration; unary reactions do not
    consume that budget. Known generated routes remain available above the bound.
    Candidates must match a catalog entry and reproduce the product forward.
    Return reverse-ordered edge lists; probability weighting belongs to the policy.
    """

    def __init__(self, env: SynthesisEnv):
        self.uni_reactions = env.uni_reactions
        self.bi_reactions = env.bi_reactions
        self.synthon_search = {
            library_name: {smiles: index for index, smiles in enumerate(library.smiles)}
            for library_name, library in env.synthons.items()
        }
        self.brick_types = set(env.brick_types)
        self._memo: dict[tuple[str, int, int, int], list[BackwardTrajectory]] = {}
        self._max_depth = 0
        self._min_synthons = 0

    def run(
        self,
        smiles: str,
        max_reactions: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        self._max_depth = max_reactions + 1  # Bound reaction cycles independently.
        self._min_synthons = max_reactions + 1  # FirstSynthon plus binary reactions.
        if known_trajectories:
            self._min_synthons = min(
                self._min_synthons,
                min(
                    sum(not action.action_type.is_unirxn for action, _ in route)
                    for route in known_trajectories
                ),
            )
        self._memo = {}
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return []
        return self._dfs(mol, Chem.MolToSmiles(mol), 1, 1, known_trajectories)

    def _dfs(
        self,
        mol: Chem.Mol,
        canonical: str,
        depth: int,
        num_synthons: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        # 1. Count removed binary reactants plus the eventual FirstSynthon.
        # UniReaction advances depth but leaves this synthon count unchanged.
        if depth > self._max_depth or num_synthons > self._min_synthons:
            return []
        # Cache only under the same remaining reaction and synthon budgets.
        key = (canonical, depth, num_synthons, self._min_synthons)
        if known_trajectories is None and key in self._memo:
            return self._memo[key]
        trajectories = list(known_trajectories or [])
        # First edges identify backward choices. Retain known suffixes and skip
        # rediscovering their action/parent pair during this root search.
        branch_keys = {trajectory[0] for trajectory in trajectories}

        # 2. Look for a direct FirstSynthon origin by restoring the catalog marker.
        signature = typed_dummy_isotopes(mol)
        if len(signature) == 1:
            synthon_type = signature[0]
            brick = Chem.Mol(mol)
            for atom in brick.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(0)
            brick_smiles = Chem.MolToSmiles(brick)
            library_name = str(synthon_type)
            if library_name in self.brick_types:
                synthon_index = self.synthon_search[library_name].get(brick_smiles)
                if synthon_index is not None:
                    self._min_synthons = num_synthons
                    action = Action(
                        ActionType.FIRST_SYNTHON,
                        library_name=library_name,
                        synthon_index=synthon_index,
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
                    action = Action(
                        ActionType.UNIRXN_TERMINAL
                        if reaction.output_type is None
                        else ActionType.UNIRXN_TRANSFORM,
                        reaction=name,
                    )
                    if (action, parent_smiles) in branch_keys:
                        continue
                    forward_product = reaction.run_forward(precursor)
                    if (
                        forward_product is None
                        or Chem.MolToSmiles(forward_product) != canonical
                    ):
                        continue
                    suffixes = self._dfs(
                        precursor, parent_smiles, depth + 1, num_synthons
                    )
                    if suffixes:
                        trajectories.extend(
                            [(action, parent_smiles), *suffix] for suffix in suffixes
                        )
                        branch_keys.add((action, parent_smiles))

            # 4. Reverse couplings, recover the oriented synthon, and find its row.
            for name, action in self.bi_reactions.items():
                if num_synthons >= self._min_synthons:
                    break
                for child_mol, synthon_mol in action.run_reverse(mol):
                    if num_synthons >= self._min_synthons:
                        break
                    child_canonical = Chem.MolToSmiles(child_mol)
                    synthon_canonical = Chem.MolToSmiles(synthon_mol)
                    if typed_dummy_isotopes(child_mol) != (action.state_type,):
                        continue
                    # Reverse products carry the incoming isotope-0 attachment
                    # and (for linkers) the remaining type. Together with the
                    # reaction's incoming type this determines one library.
                    synthon_sites = typed_dummy_isotopes(synthon_mol)
                    if not synthon_sites or synthon_sites[0] != 0:
                        continue
                    library_name = "-".join(
                        map(str, (action.attachment_type, *synthon_sites[1:]))
                    )
                    library = self.synthon_search.get(library_name)
                    if library is None:
                        continue
                    synthon_index = library.get(synthon_canonical)
                    if synthon_index is None:
                        continue
                    reverse_action = Action(
                        ActionType.BIRXN_BRICK
                        if len(synthon_sites) == 1
                        else ActionType.BIRXN_LINKER,
                        reaction=name,
                        library_name=library_name,
                        synthon_index=synthon_index,
                    )
                    if (reverse_action, child_canonical) in branch_keys:
                        continue
                    forward_product = action.run_forward(child_mol, synthon_mol)
                    if (
                        forward_product is None
                        or Chem.MolToSmiles(forward_product) != canonical
                    ):
                        continue
                    suffixes = self._dfs(
                        child_mol, child_canonical, depth + 1, num_synthons + 1
                    )
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
