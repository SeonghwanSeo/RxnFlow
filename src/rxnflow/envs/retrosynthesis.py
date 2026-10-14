"""Retrosynthetic route enumeration for synthon trajectories."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from rxnflow.core.molecule import Molecule
from rxnflow.core.synthon import typed_dummy_isotopes
from rxnflow.core.types import Action, ActionType, BackwardTrajectory

if TYPE_CHECKING:
    from rxnflow.core.reaction import BiReaction, UniReaction
    from rxnflow.envs.env import SynthesisEnv


_HYDROGEN_ATOM = Chem.MolFromSmarts("[#1]")


class Worker:
    """Find catalog-supported routes within the supplied reaction and synthon limits.

    The fewest synthons found bounds further exploration; unimolecular reactions do not
    consume that budget. Known generated routes remain available above this adaptive
    bound, but must satisfy the supplied reaction and synthon limits.
    Candidates must match a catalog entry and reproduce the product forward.
    Return reverse-ordered edge lists; probability weighting belongs to the policy.
    """

    def __init__(self, env: SynthesisEnv, prune: bool = True):
        # DFS setting
        self.prune: bool = prune
        # The environment's generation limits
        self.max_reactions: int = env.max_reactions
        self.max_synthons: int = env.max_synthons
        # The environment's reaction and synthon definitions
        self.uni_reactions: dict[str, UniReaction] = env.uni_reactions
        self.bi_reactions: dict[str, BiReaction] = env.bi_reactions
        self.brick_types = set(env.brick_types)
        self.synthon_search = {
            library_name: {smiles: index for index, smiles in enumerate(library.smiles)}
            for library_name, library in env.synthons.items()
        }
        self._uni_candidates: dict[
            tuple[int, ...], list[tuple[str, UniReaction, Action]]
        ] = {}
        for name, reaction in env.uni_reactions.items():
            signature = () if reaction.output_type is None else (reaction.output_type,)
            action = Action(
                ActionType.UNIRXN_TERMINAL
                if reaction.output_type is None
                else ActionType.UNIRXN_TRANSFORM,
                reaction=name,
            )
            self._uni_candidates.setdefault(signature, []).append(
                (name, reaction, action)
            )
        self._bi_actions = {
            subspace.name
            for space in env.reaction_action_spaces.values()
            for subspace in space
            if subspace.action_type.is_birxn
        }
        signatures = [(), *((site,) for site in sorted(env.synthon_types))]
        self._bi_candidates: dict[tuple[int, ...], list[tuple[str, BiReaction]]] = {
            signature: [
                (name, reaction)
                for name, reaction in self.bi_reactions.items()
                if (
                    name,
                    "-".join(map(str, (reaction.attachment_type, *signature))),
                )
                in self._bi_actions
            ]
            for signature in signatures
        }

    def run(
        self,
        smiles: str,
        *,
        max_reactions: int | None = None,
        max_synthons: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        if max_reactions is None:
            max_reactions = self.max_reactions
        if max_synthons is None:
            max_synthons = self.max_synthons
        if smiles is None:
            return []
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            return []

        dfs = _RetroDFS(self, max_reactions, max_synthons, prune=self.prune)
        mol = Molecule(rdmol=rdmol)
        return dfs.run(mol, known_trajectories)


class _RetroDFS:
    """Own the bounds and caches for one molecule's retrosynthetic search."""

    def __init__(
        self,
        worker: Worker,
        max_reactions: int,
        max_synthons: int,
        prune: bool = True,
    ):
        self.prune: int = prune
        self.max_depth = max_reactions + 1  # Includes FirstSynthon.
        self.brick_types = worker.brick_types
        self.synthon_search = worker.synthon_search
        self.uni_candidates = worker._uni_candidates
        self.bi_candidates = worker._bi_candidates
        self.bi_actions = worker._bi_actions
        # Reverse chemistry is independent of DFS budgets and known routes.
        self.cache: dict[str, dict] = {"unirxn": {}, "birxn": {}, "traj": {}}

        # Dynamic synthon limit
        self._max_synthons = min(max_synthons, self.max_depth)

    def run(
        self,
        molecule: Molecule,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        if known_trajectories is not None:
            # Filter known routes to those within the current limits
            known_trajectories = [
                route
                for route in known_trajectories
                if len(route) <= self.max_depth
                and sum(not action.action_type.is_unirxn for action, _ in route)
                <= self._max_synthons
            ]
            # Set the dynamic synthon limit
            if self.prune and known_trajectories:
                self._max_synthons = min(
                    self._max_synthons,
                    min(
                        sum(not action.action_type.is_unirxn for action, _ in route)
                        for route in known_trajectories
                    ),
                )
        return self._dfs(molecule, 1, 1, known_trajectories)

    def _dfs(
        self,
        molecule: Molecule,
        depth: int,
        num_synthons: int,
        known_trajectories: list[BackwardTrajectory] | None = None,
    ) -> list[BackwardTrajectory]:
        # 1. Count removed bimolecular reactants plus the eventual FirstSynthon.
        # UniReaction advances depth but leaves this synthon count unchanged.
        if depth > self.max_depth or num_synthons > self._max_synthons:
            return []
        # Cache only under the same remaining reaction and synthon budgets.
        canonical = molecule.smiles
        key = (canonical, depth, num_synthons, self._max_synthons)
        if known_trajectories is None and key in self.cache["traj"]:
            return self.cache["traj"][key]
        trajectories = list(known_trajectories or [])
        # First edges identify backward choices. Retain known suffixes and skip
        # rediscovering their action/parent pair during this root search.
        branch_keys = {trajectory[0] for trajectory in trajectories}

        # 2. Look for a direct FirstSynthon origin by restoring the catalog marker.
        signature = typed_dummy_isotopes(molecule.rdmol)
        if len(signature) == 1:
            library_name = str(signature[0])
            if library_name in self.brick_types:
                brick = Chem.Mol(molecule.rdmol)
                for atom in brick.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetIsotope(0)
                brick_smiles = Chem.MolToSmiles(brick)
                synthon_index = self.synthon_search[library_name].get(brick_smiles)
                if synthon_index is not None:
                    if self.prune and num_synthons < self._max_synthons:
                        self._max_synthons = num_synthons
                    action = Action(
                        ActionType.FIRST_SYNTHON,
                        library_name=library_name,
                        synthon_index=synthon_index,
                    )
                    edge = (action, "")
                    if edge not in branch_keys:
                        trajectories.append([edge])
                        branch_keys.add(edge)

        # 3. Reverse unimolecular transformations and verify each precursor forward.
        if depth < self.max_depth:
            for name, reaction, action in self.uni_candidates.get(signature, ()):
                reverse_key = (name, canonical)
                products_list = self.cache["unirxn"].get(reverse_key)
                if products_list is None:
                    products_list = reaction.run_reverse(molecule)
                    self.cache["unirxn"][reverse_key] = products_list
                for products in products_list:
                    if len(products) != 1:
                        continue
                    precursor = products[0]
                    parent_smiles = precursor.smiles
                    if typed_dummy_isotopes(precursor.rdmol) != (reaction.input_type,):
                        continue
                    edge = (action, parent_smiles)
                    if edge in branch_keys:
                        continue
                    forward_product = reaction.run_forward(precursor)
                    if forward_product is None or forward_product.smiles != canonical:
                        continue
                    suffixes = self._dfs(precursor, depth + 1, num_synthons)
                    if suffixes:
                        trajectories.extend([edge, *suffix] for suffix in suffixes)
                        branch_keys.add(edge)

            # 4. Reverse couplings, recover the oriented synthon, and find its row.
            for name, reaction in self.bi_candidates.get(signature, ()):
                if num_synthons >= self._max_synthons:
                    break
                reverse_key = (name, canonical)
                products_list = self.cache["birxn"].get(reverse_key)
                if products_list is None:
                    products_list = self._reverse_birxn(reaction, molecule)
                    self.cache["birxn"][reverse_key] = products_list
                for child, synthon in products_list:
                    if num_synthons >= self._max_synthons:
                        break
                    if typed_dummy_isotopes(child.rdmol) != (reaction.state_type,):
                        continue
                    # Reverse products carry the incoming isotope-0 attachment
                    # and (for linkers) the remaining type. Together with the
                    # reaction's incoming type this determines one library.
                    synthon_sites = typed_dummy_isotopes(synthon.rdmol)
                    if not synthon_sites or synthon_sites[0] != 0:
                        continue
                    library_name = "-".join(
                        map(str, (reaction.attachment_type, *synthon_sites[1:]))
                    )
                    if (name, library_name) not in self.bi_actions:
                        continue
                    library = self.synthon_search.get(library_name)
                    if library is None:
                        continue
                    synthon_index = library.get(synthon.smiles)
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
                    edge = (reverse_action, child.smiles)
                    if edge in branch_keys:
                        continue
                    forward_product = reaction.run_forward(child, synthon)
                    if forward_product is None or forward_product.smiles != canonical:
                        continue
                    suffixes = self._dfs(child, depth + 1, num_synthons + 1)
                    if suffixes:
                        trajectories.extend([edge, *suffix] for suffix in suffixes)
                        branch_keys.add(edge)

        # 5. Cache only unseeded searches; known routes are specific to a rollout.
        if known_trajectories is None:
            self.cache["traj"][key] = trajectories
        return trajectories

    def _reverse_birxn(
        self, reaction: BiReaction, product: Molecule
    ) -> list[tuple[Molecule, Molecule]]:
        """Process catalog-supported precursor pairs in canonical DFS order."""
        products = {}
        for raw in reaction.reverse_reaction.RunReactants((product.rdmol,), 0):
            if len(raw) != 2 or typed_dummy_isotopes(raw[0]) != (reaction.state_type,):
                continue
            sites = typed_dummy_isotopes(raw[1])
            if not sites or sites[0] != 0:
                continue
            library_name = "-".join(map(str, (reaction.attachment_type, *sites[1:])))
            library = self.synthon_search.get(library_name)
            if library is None:
                continue

            # Reject missing catalog synthons before sanitizing the parent.
            molecules, keys = [None, None], [None, None]
            for index in (1, 0):
                molecule = raw[index]
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(molecule)
                        if molecule.HasSubstructMatch(_HYDROGEN_ATOM):
                            molecule = Chem.RemoveHs(molecule)
                        smiles = Chem.MolToSmiles(molecule)
                except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                    break
                if index == 1 and smiles not in library:
                    break
                molecules[index], keys[index] = molecule, smiles
            if any(key is None for key in keys):
                continue
            key = tuple(keys)
            # Allocate wrappers only for complete, unique precursor pairs.
            if key not in products:
                products[key] = (
                    Molecule(smiles=keys[0], rdmol=molecules[0]),
                    Molecule(smiles=keys[1], rdmol=molecules[1]),
                )
        return [products[key] for key in sorted(products)]


_WORKER: Worker | None = None


def _init_worker(worker: Worker) -> None:
    global _WORKER
    _WORKER = worker


def _worker_run(
    smiles: str,
    max_reactions: int | None,
    known_trajectories: list[BackwardTrajectory] | None,
    max_synthons: int | None,
) -> list[BackwardTrajectory]:
    assert _WORKER is not None
    return _WORKER.run(
        smiles,
        max_reactions=max_reactions,
        known_trajectories=known_trajectories,
        max_synthons=max_synthons,
    )


class RetroSynthesisAnalyzer:
    """Find routes for external SMILES or generated states, locally or in workers.

    Search limits default to the environment's generation maxima and can be overridden
    per call.
    Results use reverse edge order, ending with FirstSynthon. Searches favor fewer
    synthons and do not enumerate every route within the limits.
    """

    def __init__(self, env: SynthesisEnv, workers: int = 0, prune: bool = True):
        self.worker = Worker(env, prune=prune)
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
        max_reactions: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
        *,
        max_synthons: int | None = None,
    ) -> list[BackwardTrajectory]:
        if self.pool is None:
            return self.worker.run(
                smiles,
                max_reactions=max_reactions,
                known_trajectories=known_trajectories,
                max_synthons=max_synthons,
            )
        return self.pool.submit(
            _worker_run, smiles, max_reactions, known_trajectories, max_synthons
        ).result()

    def submit(
        self,
        key: int,
        smiles: str,
        max_reactions: int | None = None,
        known_trajectories: list[BackwardTrajectory] | None = None,
        *,
        max_synthons: int | None = None,
    ) -> None:
        if self.pool is None:
            self.results.append(
                (
                    key,
                    self.worker.run(
                        smiles,
                        max_reactions=max_reactions,
                        known_trajectories=known_trajectories,
                        max_synthons=max_synthons,
                    ),
                )
            )
        else:
            future = self.pool.submit(
                _worker_run, smiles, max_reactions, known_trajectories, max_synthons
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
