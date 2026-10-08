"""Linear synthesis in synthon space: one growing intermediate, no Stop action."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

from rdkit.Chem.rdChemReactions import ReactionToSmarts

from rxnflow.core.compatibility import check_library_compatibility
from rxnflow.core.errors import InvalidTransition
from rxnflow.core.molecule import Molecule
from rxnflow.core.reaction import load_reactions
from rxnflow.core.synthon import load_synthon_templates, typed_dummy_isotopes
from rxnflow.core.types import Action, ActionSpace, ActionSubspace, ActionType, State
from rxnflow.envs.features import PROPERTY_PENALTY_INDICES, parse_molecule
from rxnflow.envs.library import load_synthon_libraries
from rxnflow.envs.retrosynthesis import RetroSynthesisAnalyzer


class SynthesisEnv:
    """Synthon-based linear synthesis environment"""

    def __init__(
        self,
        env_dir: str | Path,
        max_atoms: int = 50,
        max_reactions: int = 3,
        min_synthons: int = 2,
        max_synthons: int = 3,
        min_reactions: int = 1,
        property_penalty: dict[str, float] | None = None,
        retrosynthesis_workers: int = 0,
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
            PROPERTY_PENALTY_INDICES[name]: limit
            for name, limit in (property_penalty or {}).items()
        }
        self.retrosynthesis_workers = retrosynthesis_workers
        self.signature = json.loads((self.env_dir / "signature.json").read_text())
        check_library_compatibility(self.signature["rxnflow_version"])
        self._load_libraries()
        self._load_reactions()
        self._load_action_spaces()
        self._build_feasible_action_spaces()

    def _load_libraries(self) -> None:
        """Load aligned synthon data and derive library/type indices."""
        self.synthons = load_synthon_libraries(self.env_dir)
        with (self.env_dir / "building_blocks.smi").open(encoding="utf-8") as handle:
            records = (line.rstrip("\n").split("\t") for line in handle)
            self.sources = {identifier: smiles for smiles, identifier in records}
        self.synthon_to_bb_ids: dict[str, list[str]] = {}
        for path in sorted((self.env_dir / "synthons").glob("*.tsv")):
            with path.open(encoding="utf-8") as handle:
                next(handle)  # synthon_id, smiles, bb_ids
                for line in handle:
                    synthon_id, _, bb_ids = line.rstrip("\n").split("\t")
                    self.synthon_to_bb_ids[synthon_id] = bb_ids.split(";")
        self.library_names = sorted(self.synthons)
        self.brick_types = [
            name for name in self.library_names if self.synthons[name].is_brick
        ]
        if not self.brick_types:
            raise ValueError("prepared environment contains no one-site bricks")

        self.synthon_types = set(load_synthon_templates(self.env_dir / "synthon.yaml"))
        # Index zero represents an absent remaining site on a brick.
        # Chemical types use stable indices independent of catalog membership.
        self.synthon_type_to_index = {
            t: i + 1 for i, t in enumerate(sorted(self.synthon_types))
        }
        self.library_site_indices = {
            name: (
                self.synthon_type_to_index[library.attachment_type],
                self.synthon_type_to_index[library.synthon_types[1]]
                if library.is_linker
                else 0,
            )
            for name, library in self.synthons.items()
        }
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

    def _build_feasible_action_spaces(self) -> None:
        """Keep type-level paths that can terminate within the synthesis limits.

        This ensure that `get_action_space` always returns a action space that
        can reach a terminal state.
        """
        self.feasible_action_spaces: dict[tuple[int, int, int], ActionSpace] = {}
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
                                    self.feasible_action_spaces.get(
                                        (next_site, next_synthons, reactions + 1)
                                    )
                                )
                            if reachable:
                                allowed.append(subspace)
                    self.feasible_action_spaces[site, synthons, reactions] = allowed
        # FirstSynthon consumes a synthon but no reaction. Exclude starting types
        # with no complete route under these limits.
        self.initial_action_space = [
            subspace
            for subspace in self.initial_action_space
            if self.feasible_action_spaces[
                self.synthons[subspace.name[1]].attachment_type, 1, 0
            ]
        ]
        if not self.initial_action_space:
            raise ValueError(
                "no synthesis path can terminate within the synthesis limits"
            )

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
        elif state.molecule is None:
            return self.initial_action_space
        assert state.num_reactions < self.max_reactions
        key = (state.attachment_type, state.num_synthons, state.num_reactions)
        return self.feasible_action_spaces[key]

    def _apply_action(self, current: Molecule | None, action: Action) -> Molecule | None:
        """Apply FirstSynthon/UniReaction/BiReaction and return a valid molecule."""
        # 1. Resolve the selected synthon row.
        if action.library_name is not None:
            library = self.synthons[action.library_name]
            if action.synthon_index is None or not 0 <= action.synthon_index < len(
                library
            ):
                raise ValueError("synthon_index is out of range")
            synthon_smiles = library.smiles[action.synthon_index]
        elif action.synthon_index is not None:
            raise ValueError("unimolecular actions do not take a synthon_index")

        # 2. Apply the transformation and determine the expected remaining sites.
        if action.action_type == ActionType.FIRST_SYNTHON:
            # Catalog attachment markers are always 0. A first brick becomes a
            # state with its chemical synthon type restored from the library.
            first = parse_molecule(synthon_smiles)
            assert first is not None
            for atom in first.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetIsotope(library.attachment_type)
            product = Molecule(rdmol=first)
            expected = library.synthon_types
        else:
            assert current is not None
            if action.action_type.is_unirxn:
                reaction = self.uni_reactions[action.reaction]
                product = reaction.run_forward(current)
                expected = () if reaction.output_type is None else (reaction.output_type,)
            else:
                bi = self.bi_reactions[action.reaction]
                synthon = parse_molecule(synthon_smiles)
                assert synthon is not None
                product = bi.run_forward(
                    current, Molecule(smiles=synthon_smiles, rdmol=synthon)
                )
                expected = library.synthon_types[1:]
        # 3. Check remaining sites and structural change. The approximate atom
        # budget affects action selection; exceeding it does not invalidate a product.
        if product is None or typed_dummy_isotopes(product.rdmol) != expected:
            return None
        # Reject transformations that leave the canonical structure unchanged.
        if current is not None and product.smiles == current.smiles:
            return None
        return product

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
        product = self._apply_action(state.molecule, action)
        if product is None:
            raise InvalidTransition("the selected reaction failed structural checks")
        count = state.num_reactions + int(action.action_type != ActionType.FIRST_SYNTHON)
        terminal = not typed_dummy_isotopes(product.rdmol)
        return State(
            molecule=product,
            num_reactions=count,
            num_synthons=state.num_synthons + int(not action.action_type.is_unirxn),
            terminated=terminal,
        )

    def action_to_dict(self, action: Action) -> dict[str, object]:
        """Export the executed template and source BB candidates for an action."""
        reaction_smarts = None
        if not action.action_type.is_first:
            reactions = (
                self.uni_reactions
                if action.action_type.is_unirxn
                else self.bi_reactions
            )
            # Include incoming-site orientation in the executable template.
            reaction_smarts = ReactionToSmarts(
                reactions[action.reaction].forward_reaction
            )
        result: dict[str, object] = {
            "action_type": action.action_type.name,
            "reaction": action.reaction,
            "reaction_smarts": reaction_smarts,
        }
        if action.library_name is not None:
            assert action.synthon_index is not None
            library = self.synthons[action.library_name]
            synthon_id = library.synthon_ids[action.synthon_index]
            identifiers = self.synthon_to_bb_ids[synthon_id]
            result["synthon_smiles"] = library.smiles[action.synthon_index]
            result["synthon_id"] = synthon_id
            result["building_blocks"] = [
                {"id": identifier, "smiles": self.sources[identifier]}
                for identifier in identifiers
            ]
        return result
