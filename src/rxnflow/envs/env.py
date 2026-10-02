"""Linear synthesis in synthon space: one growing intermediate, no Stop action."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from rxnflow.core.errors import InvalidTransition
from rxnflow.core.reaction import load_reactions
from rxnflow.core.synthon import load_synthon_specs, typed_dummy_isotopes
from rxnflow.core.types import Action, ActionSpace, ActionSubspace, ActionType, State
from rxnflow.envs.features import (
    PROPERTY_NAMES,
    heavy_atom_count,
    parse_molecule,
)
from rxnflow.envs.library import load_synthon_libraries
from rxnflow.envs.retrosynthesis import RetroSynthesisAnalyzer


class SynthesisEnv:
    """Grow one synthon intermediate through typed unary and binary reactions."""

    def __init__(
        self,
        env_dir: str | Path,
        max_atoms: int = 50,
        max_reactions: int = 3,
        retrosynthesis_workers: int = 0,
        property_penalty: dict[str, float] | None = None,
        *,
        min_synthons: int = 2,
        max_synthons: int = 3,
        min_reactions: int = 1,
    ):
        self.env_dir = Path(env_dir)
        self.max_atoms = max_atoms
        self.min_synthons = min_synthons
        self.max_synthons = max_synthons
        self.min_reactions = min_reactions
        self.max_reactions = max_reactions
        if (
            not 1 <= min_synthons <= max_synthons
            or not 1 <= min_reactions <= max_reactions
        ):
            raise ValueError("invalid synthon/reaction bounds")
        if min_synthons > max_reactions + 1:
            raise ValueError("min_synthons cannot be reached within max_reactions")
        if max_atoms <= 0 or max_reactions < 1:
            raise ValueError("invalid synthesis environment limits")
        self.property_limits = {
            PROPERTY_NAMES.index(name): limit
            for name, limit in (property_penalty or {}).items()
        }
        self.retrosynthesis_workers = retrosynthesis_workers
        self._load_libraries()
        self._load_reactions()
        self._load_action_spaces()
        self._build_budget_action_spaces()

    def _load_libraries(self) -> None:
        """Load aligned synthon data and derive library/type indices."""
        self.synthons = load_synthon_libraries(self.env_dir)
        self.sources = json.loads((self.env_dir / "building_blocks.json").read_text())
        self.library_names = sorted(self.synthons)
        self.library_to_index = {name: i for i, name in enumerate(self.library_names)}
        self.brick_types = [
            name for name in self.library_names if self.synthons[name].is_brick
        ]
        if not self.brick_types:
            raise ValueError("prepared environment contains no one-site bricks")

        specs = load_synthon_specs(self.env_dir / "synthon.yaml")
        self.synthon_types = {spec.type for spec in specs}
        for library in self.synthons.values():
            if (
                len(library.synthon_types) not in (1, 2)
                or not set(library.synthon_types) <= self.synthon_types
            ):
                raise ValueError(f"invalid brick/linker type: {library.name}")

    def _load_reactions(self) -> None:
        """Compile oriented reactions and assign their policy indices."""
        self.uni_reactions, self.bi_reactions = load_reactions(
            self.env_dir / "reaction.yaml"
        )
        for reaction in self.uni_reactions.values():
            if reaction.input_type not in self.synthon_types or (
                reaction.output_type is not None
                and reaction.output_type not in self.synthon_types
            ):
                raise ValueError(f"unknown synthon type in {reaction.name}")
        for reaction in self.bi_reactions.values():
            if not set(reaction.synthon_types) <= self.synthon_types:
                raise ValueError(f"unknown synthon type in {reaction.name}")
        self.action_names = [
            "first_synthon",
            *sorted(self.uni_reactions),
            *sorted(self.bi_reactions),
        ]
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("reaction action names must be unique")
        self.action_to_index = {name: i for i, name in enumerate(self.action_names)}

    def _load_action_spaces(self) -> None:
        """Connect prepared reaction/library pairs to the loaded runtime objects."""
        spaces = json.loads((self.env_dir / "action_space.json").read_text())

        def load_space(pairs: list[list[str | None]]) -> ActionSpace:
            result = []
            for reaction, library in pairs:
                if reaction == "first_synthon":
                    action_type = ActionType.FIRST_SYNTHON
                elif library is None:
                    action_type = (
                        ActionType.UNIRXN_TERMINAL
                        if self.uni_reactions[reaction].output_type is None
                        else ActionType.UNIRXN_TRANSFORM
                    )
                else:
                    action_type = (
                        ActionType.BIRXN_BRICK
                        if self.synthons[library].is_brick
                        else ActionType.BIRXN_LINKER
                    )
                result.append(
                    ActionSubspace(
                        (reaction, library),
                        action_type,
                        1 if library is None else len(self.synthons[library]),
                    )
                )
            return result

        self.initial_action_space = load_space(spaces["initial"])
        self.reaction_action_spaces = {
            int(site): load_space(pairs) for site, pairs in spaces["reaction"].items()
        }
        # Prepared artifacts are immutable until the next preparation. Reading
        # this saved signature avoids hashing the full feature archive at startup.
        self.signature = json.loads((self.env_dir / "signature.json").read_text())

    def _build_budget_action_spaces(self) -> None:
        """Keep type-level transitions that can finish within both budgets.

        Work backwards in reaction count: every Uni/Bi transition consumes one
        reaction, so each successor has already been computed. This is chemistry
        type feasibility; property penalties still apply at sampling time.
        """
        self.budget_action_spaces: dict[tuple[int, int, int], ActionSpace] = {}
        for reactions in range(self.max_reactions, -1, -1):
            for synthons in range(1, self.max_synthons + 1):
                for site, space in self.reaction_action_spaces.items():
                    allowed = []
                    if reactions < self.max_reactions:
                        for subspace in space:
                            reaction, library = subspace.name
                            next_synthons = synthons + int(library is not None)
                            if next_synthons > self.max_synthons:
                                continue
                            if library is None:
                                next_site = self.uni_reactions[reaction].output_type
                            else:
                                sites = self.synthons[library].synthon_types
                                next_site = None if len(sites) == 1 else sites[1]
                            if next_site is None:
                                reachable = (
                                    next_synthons >= self.min_synthons
                                    and reactions + 1 >= self.min_reactions
                                )
                            else:
                                reachable = bool(
                                    self.budget_action_spaces.get(
                                        (next_site, next_synthons, reactions + 1)
                                    )
                                )
                            if reachable:
                                allowed.append(subspace)
                    self.budget_action_spaces[site, synthons, reactions] = allowed
        # FirstSynthon consumes a synthon but no reaction. Exclude starting types
        # with no complete route under these limits.
        self.initial_action_space = [
            subspace
            for subspace in self.initial_action_space
            if self.budget_action_spaces[
                self.synthons[subspace.name[1]].attachment_type, 1, 0
            ]
        ]

    @cached_property
    def retro_analyzer(self) -> RetroSynthesisAnalyzer:
        return RetroSynthesisAnalyzer(self, self.retrosynthesis_workers)

    def close(self) -> None:
        """Release reverse-search workers without creating an unused analyzer."""
        analyzer = self.__dict__.pop("retro_analyzer", None)
        if analyzer is not None:
            analyzer.close()

    @staticmethod
    def initial_state() -> State:
        return State()

    @staticmethod
    def is_terminal(state: State) -> bool:
        return state.terminated

    @staticmethod
    def get_synthon_types(smiles: str) -> tuple[int, ...]:
        """Return the synthon types encoded by dummy isotopes in SMILES."""
        mol = parse_molecule(smiles)
        return () if mol is None else typed_dummy_isotopes(mol)

    def get_action_space(self, state: State) -> ActionSpace:
        """Look up type/step eligibility; property penalties follow subsampling."""
        if state.terminated:
            return []
        if state.mol is None:
            return self.initial_action_space
        if state.num_reactions >= self.max_reactions:
            return []
        signature = typed_dummy_isotopes(state.mol)
        if len(signature) != 1 or signature[0] not in self.synthon_types:
            return []
        return self.budget_action_spaces.get(
            (signature[0], state.num_synthons, state.num_reactions), []
        )

    def get_synthon_mask(
        self,
        state_properties: NDArray[np.float32],
        library_name: str,
        indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.bool_]:
        """Combine the atom-capacity action mask with the property penalty.

        Inputs are one state [P] or a batch [B, P]; the result is [N] or [B, N].
        Compare raw units so zero and negative upper bounds remain meaningful.
        These sums estimate product properties without executing candidate reactions.
        """
        library = self.synthons[library_name]
        heavy_atoms = (
            library.heavy_atoms if indices is None else library.heavy_atoms[indices]
        )
        properties = (
            library.properties if indices is None else library.properties[indices]
        )
        # Broadcast either one state [P] or a state batch [B, P] against
        # the common sampled library [N]. This gathers synthon features once.
        # State properties are float32, so uint8 counts are promoted before
        # addition; a combined count above 255 cannot wrap.
        mask = (
            heavy_atoms + state_properties[..., PROPERTY_NAMES.index("heavy_atoms"), None]
            <= self.max_atoms
        )
        if self.property_limits:
            property_penalty = self.compute_property_penalty(state_properties, properties)
            mask &= property_penalty
        return mask

    def compute_property_penalty(
        self,
        state_properties: NDArray[np.float32],
        synthon_properties: NDArray[np.float32],
    ) -> NDArray[np.bool_]:
        """Return binary Ω: True (1) permits an action; False (0) excludes it.

        Use additive state + synthon estimates, without executing reactions.
        The result is [N] or [B, N]. The max_atoms action mask is applied separately.
        """
        property_penalty = np.ones(
            (*state_properties.shape[:-1], len(synthon_properties)), dtype=np.bool_
        )
        for index, limit in self.property_limits.items():
            estimate = synthon_properties[:, index] + state_properties[..., index, None]
            # Nonzero bounds allow 1% of their magnitude as tolerance.
            # Zero bounds stay exact.
            if limit == 0:
                property_penalty &= estimate <= 0
            else:
                property_penalty &= estimate < limit + abs(limit) * 0.01
        return property_penalty

    def _apply_action(self, current: Chem.Mol | None, action: Action) -> Chem.Mol | None:
        """Apply FirstSynthon/UniReaction/BiReaction and return a valid Mol or None."""
        # 1. Resolve the selected catalog row, when this action consumes a synthon.
        if action.library_name is not None:
            library = self.synthons[action.library_name]
            if action.synthon_index is None or not 0 <= action.synthon_index < len(
                library
            ):
                raise ValueError("synthon_index is out of range")
            synthon_smiles = library.smiles[action.synthon_index]
        elif action.synthon_index is not None:
            raise ValueError("unary actions do not take a synthon_index")

        # 2. Execute the selected transformation and determine its expected handle.
        if action.action_type == ActionType.FIRST_SYNTHON:
            # Catalog attachment markers are always 0. A first brick becomes a
            # state with its chemical synthon type restored from the library.
            first = parse_molecule(synthon_smiles)
            assert first is not None
            for atom in first.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(library.attachment_type)
            mol = first
            expected = library.synthon_types
        else:
            assert current is not None
            if action.action_type.is_unirxn:
                reaction = self.uni_reactions[action.reaction]
                mol = reaction.run_forward(current)
                expected = () if reaction.output_type is None else (reaction.output_type,)
            else:
                bi = self.bi_reactions[action.reaction]
                synthon = parse_molecule(synthon_smiles)
                assert synthon is not None
                mol = bi.run_forward(current, synthon)
                expected = library.synthon_types[1:]
        # 3. Check the actual product's handle, capacity, and structural change.
        if mol is None or typed_dummy_isotopes(mol) != expected:
            return None
        if heavy_atom_count(mol) > self.max_atoms:
            return None
        # Canonical SMILES detect no-ops; retain the molecule itself so masking,
        # the selected transition, and model features share the same product.
        if current is not None and Chem.MolToSmiles(mol) == Chem.MolToSmiles(current):
            return None
        return mol

    def step(self, state: State, action: Action) -> State:
        """Execute an action and terminate when its product has no marked handle."""
        name = (
            "first_synthon"
            if action.action_type == ActionType.FIRST_SYNTHON
            else action.reaction,
            action.library_name,
        )
        if not any(
            space.name == name and space.action_type == action.action_type
            for space in self.get_action_space(state)
        ):
            raise InvalidTransition("action cannot finish within the synthesis budgets")
        product = self._apply_action(state.mol, action)
        if product is None:
            raise InvalidTransition(
                "the selected reaction failed structural or graph-capacity checks"
            )
        count = state.num_reactions + int(action.action_type != ActionType.FIRST_SYNTHON)
        terminal = not typed_dummy_isotopes(product)
        return State(
            mol=product,
            num_reactions=count,
            num_synthons=state.num_synthons + int(not action.action_type.is_unirxn),
            terminated=terminal,
        )

    def action_to_dict(self, action: Action) -> dict[str, object]:
        result = action.to_dict()
        if action.library_name is not None:
            assert action.synthon_index is not None
            library = self.synthons[action.library_name]
            identifiers = library.identifiers[action.synthon_index]
            result["synthon_smiles"] = library.smiles[action.synthon_index]
            result["synthon_ids"] = identifiers
            result["building_blocks"] = [
                {"id": identifier, "smiles": self.sources[identifier]}
                for identifier in identifiers
            ]
        return result
