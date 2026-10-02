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
from rxnflow.envs.library import load_block_libraries
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
    ):
        self.env_dir = Path(env_dir)
        self.max_atoms = max_atoms
        self.max_reactions = max_reactions
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

    def _load_libraries(self) -> None:
        """Load aligned block data and derive library/type indices."""
        self.blocks = load_block_libraries(self.env_dir)
        self.sources = json.loads((self.env_dir / "building_blocks.json").read_text())
        self.block_types = sorted(self.blocks)
        self.block_type_to_index = {name: i for i, name in enumerate(self.block_types)}
        self.brick_types = [
            name for name in self.block_types if self.blocks[name].is_brick
        ]
        if not self.brick_types:
            raise ValueError("prepared environment contains no one-site bricks")

        specs = load_synthon_specs(self.env_dir / "synthon.yaml")
        self.synthon_types = {spec.type for spec in specs}
        for library in self.blocks.values():
            if (
                len(library.site_types) not in (1, 2)
                or not set(library.site_types) <= self.synthon_types
            ):
                raise ValueError(f"invalid brick/linker type: {library.block_type}")

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
            if not set(reaction.block_types) <= self.synthon_types:
                raise ValueError(f"unknown synthon type in {reaction.name}")
        self.action_names = [
            "first_block",
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
                if reaction == "first_block":
                    action_type = ActionType.FIRST_BLOCK
                elif library is None:
                    action_type = ActionType.UNI_REACTION
                else:
                    action_type = ActionType.BI_REACTION
                result.append(
                    ActionSubspace(
                        (reaction, library),
                        action_type,
                        1 if library is None else len(self.blocks[library]),
                    )
                )
            return result

        self.initial_action_space = load_space(spaces["initial"])
        self.reaction_action_spaces = {
            int(site): load_space(pairs) for site, pairs in spaces["reaction"].items()
        }
        # Last-step lists refer to the same subspaces, not duplicated objects.
        by_name = {
            subspace.name: subspace
            for space in self.reaction_action_spaces.values()
            for subspace in space
        }
        self.last_action_spaces = {
            int(site): [by_name[tuple(pair)] for pair in pairs]
            for site, pairs in spaces["last"].items()
        }
        # Derived from connected spaces, so counts cannot drift from libraries.
        # This is a branching scale for backward weights, not a state action count.
        self.num_total_actions = max(
            2,
            len(self.uni_reactions)
            + sum(subspace.num_actions for subspace in self.initial_action_space)
            + sum(
                subspace.num_actions
                for space in self.reaction_action_spaces.values()
                for subspace in space
                if subspace.action_type == ActionType.BI_REACTION
            ),
        )
        # Prepared artifacts are immutable until the next preparation. Reading
        # this saved signature avoids hashing the full feature archive at startup.
        self.signature = json.loads((self.env_dir / "signature.json").read_text())

    @cached_property
    def retro_analyzer(self) -> RetroSynthesisAnalyzer:
        return RetroSynthesisAnalyzer(self, self.retrosynthesis_workers)

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
        """Look up type/step eligibility; property masks follow subsampling."""
        if state.terminated:
            return []
        if state.mol is None:
            return self.initial_action_space
        if state.reaction_count >= self.max_reactions:
            return []
        signature = typed_dummy_isotopes(state.mol)
        if len(signature) != 1 or signature[0] not in self.synthon_types:
            return []
        if state.reaction_count + 1 == self.max_reactions:
            return self.last_action_spaces[signature[0]]
        return self.reaction_action_spaces[signature[0]]

    def get_block_mask(
        self,
        state_properties: NDArray[np.float32],
        block_type: str,
        indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.bool_]:
        """Mask block rows using additive state + block property estimates.

        Inputs are one state [P] or a batch [B, P]; the result is [N] or [B, N].
        Compare raw units so zero and negative upper bounds remain meaningful.
        These sums estimate product properties without executing candidate reactions.
        """
        library = self.blocks[block_type]
        heavy_atoms = (
            library.heavy_atoms if indices is None else library.heavy_atoms[indices]
        )
        properties = (
            library.properties if indices is None else library.properties[indices]
        )
        # Broadcast either one state [P] or a state batch [B, P] against
        # the common sampled library [N]. This gathers block features once.
        # State properties are float32, so uint8 counts are promoted before
        # addition; a combined count above 255 cannot wrap.
        mask = (
            heavy_atoms + state_properties[..., PROPERTY_NAMES.index("heavy_atoms"), None]
            <= self.max_atoms
        )
        for index, limit in self.property_limits.items():
            estimate = properties[:, index] + state_properties[..., index, None]
            # Nonzero bounds allow 1% of their magnitude as tolerance.
            # Zero bounds stay exact; the atom-capacity check above is strict.
            if limit == 0:
                mask &= estimate <= 0
            else:
                mask &= estimate < limit + abs(limit) * 0.01
        return mask

    def _apply_action(self, current: Chem.Mol | None, action: Action) -> Chem.Mol | None:
        """Apply FirstBlock/UniReaction/BiReaction and return a valid Mol or None."""
        # 1. Resolve the selected catalog row, when this action consumes a block.
        if action.block_type is not None:
            library = self.blocks[action.block_type]
            if action.block_index is None or not 0 <= action.block_index < len(library):
                raise ValueError("block_index is out of range")
            block_smiles = library.smiles[action.block_index]
        elif action.block_index is not None:
            raise ValueError("unary actions do not take a block_index")

        # 2. Execute the selected transformation and determine its expected handle.
        if action.action_type == ActionType.FIRST_BLOCK:
            # Catalog attachment markers are always 0. A first brick becomes a
            # state with its chemical synthon type restored from the library.
            first = parse_molecule(block_smiles)
            assert first is not None
            for atom in first.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(library.attachment_type)
            mol = first
            expected = library.site_types
        else:
            assert current is not None
            if action.action_type == ActionType.UNI_REACTION:
                reaction = self.uni_reactions[action.reaction]
                mol = reaction.run_forward(current)
                expected = () if reaction.output_type is None else (reaction.output_type,)
            else:
                bi = self.bi_reactions[action.reaction]
                block = parse_molecule(block_smiles)
                assert block is not None
                mol = bi.run_forward(current, block)
                expected = library.site_types[1:]
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
        product = self._apply_action(state.mol, action)
        if product is None:
            raise InvalidTransition(
                "the selected reaction failed structural or graph-capacity checks"
            )
        count = state.reaction_count + int(action.action_type != ActionType.FIRST_BLOCK)
        terminal = not typed_dummy_isotopes(product)
        return State(product, count, terminal)

    def action_to_dict(self, action: Action) -> dict[str, object]:
        result = action.to_dict()
        if action.block_type is not None:
            assert action.block_index is not None
            library = self.blocks[action.block_type]
            identifiers = library.identifiers[action.block_index]
            result["block_type"] = "brick" if library.is_brick else "linker"
            result["block_smiles"] = library.smiles[action.block_index]
            result["block_ids"] = identifiers
            result["building_blocks"] = [
                {"id": identifier, "smiles": self.sources[identifier]}
                for identifier in identifiers
            ]
        return result
