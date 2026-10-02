"""Linear synthesis in synthon space: one growing intermediate, no Stop action."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property, lru_cache
from pathlib import Path

import torch
import yaml
from rdkit import Chem

from rxnflow.envs.chemistry.features import (
    PROPERTY_NAMES,
    heavy_atom_count,
    parse_molecule,
)
from rxnflow.envs.chemistry.reaction import BiReaction, UniReaction
from rxnflow.envs.chemistry.synthon import load_synthon_specs, typed_dummy_isotopes
from rxnflow.errors import InvalidTransition
from rxnflow.gflownet.types import Action, ActionSpace, ActionType, State

from .library import load_block_libraries


class SynthesisEnv:
    """Catalog bricks have one handle; linkers have two, including latent ones.

    FirstBlock does not count as a reaction. Every Uni/BiReaction counts once.
    A reaction product with no handle is immediately terminal. The final allowed
    reaction must close the remaining handle; there is no restoration/capping.
    """

    def __init__(
        self,
        env_dir: str | Path,
        max_atoms: int = 50,
        min_reactions: int = 1,
        max_reactions: int = 3,
        retrosynthesis_workers: int = 0,
        property_penalty: dict[str, float] | None = None,
    ):
        self.env_dir = Path(env_dir)
        self.max_atoms = max_atoms
        self.min_reactions = min_reactions
        self.max_reactions = max_reactions
        if max_atoms <= 0 or min_reactions < 0 or max_reactions < max(1, min_reactions):
            raise ValueError("invalid synthesis environment limits")
        self.property_limits = {
            PROPERTY_NAMES.index(name): limit
            for name, limit in (property_penalty or {}).items()
        }
        self.blocks = load_block_libraries(self.env_dir)
        self.sources = json.loads((self.env_dir / "building_blocks.json").read_text())
        self.block_types = sorted(self.blocks)
        self.block_type_to_index = {name: i for i, name in enumerate(self.block_types)}
        self.blocks_by_attachment: dict[int, list[str]] = {}
        for name, library in self.blocks.items():
            self.blocks_by_attachment.setdefault(library.attachment_type, []).append(name)
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
        self.uni_reactions, self.bi_reactions = self._load_reactions(
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
        # A reference branching scale for the bounded backward heuristic. Site
        # outcomes are state dependent, so this is not an exact action count.
        self.num_total_actions = max(
            2,
            len(self.uni_reactions)
            + sum(len(self.blocks[name]) for name in self.brick_types)
            + sum(
                len(self.blocks[name])
                for action in self.bi_reactions.values()
                for name in self._compatible_block_types(action.block_type)
            ),
        )
        # The full static space depends only on the current handle (None at
        # initialization). Specs contain no sampled block indices or tensors.
        self.action_space: dict[int | None, ActionSpace] = {
            None: [
                (ActionType.FIRST_BLOCK, "first_block", name) for name in self.brick_types
            ],
            **{site_type: [] for site_type in self.synthon_types},
        }
        terminal_specs = set()
        for name, reaction in self.uni_reactions.items():
            spec = (ActionType.UNI_REACTION, name, None)
            self.action_space[reaction.input_type].append(spec)
            if reaction.output_type is None:
                terminal_specs.add(spec)
        for name, reaction in self.bi_reactions.items():
            for block_type in self._compatible_block_types(reaction.block_type):
                spec = (ActionType.BI_REACTION, name, block_type)
                self.action_space[reaction.state_type].append(spec)
                if self.blocks[block_type].is_brick:
                    terminal_specs.add(spec)

        # Step restrictions also depend only on static metadata. Build the
        # four variants once; states reuse these lists without scanning libraries.
        self._step_action_space: dict[tuple[int, bool, bool], ActionSpace] = {}
        for site_type in self.synthon_types:
            for last_step in (False, True):
                for may_terminate in (False, True):
                    self._step_action_space[site_type, last_step, may_terminate] = [
                        spec
                        for spec in self.action_space[site_type]
                        if (not last_step or spec in terminal_specs)
                        and (may_terminate or spec not in terminal_specs)
                    ]
        self.signature = self._build_signature()
        # Chemistry is independent of trajectory length; replay can reuse it.
        self._product = lru_cache(maxsize=8192)(self._reaction_product)

        self.retrosynthesis_workers = retrosynthesis_workers

    @cached_property
    def retro_analyzer(self):
        # Generation does not use backward probabilities. Build its catalog
        # search index and workers only when training/backward analysis needs it.
        from .retrosynthesis import RetrosynthesisSearch, RetrosynthesisWorkers

        return RetrosynthesisWorkers(
            RetrosynthesisSearch(self), self.retrosynthesis_workers
        )

    @staticmethod
    def _load_reactions(
        path: Path,
    ) -> tuple[dict[str, UniReaction], dict[str, BiReaction]]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"UniReaction", "BiReaction"}:
            raise ValueError(
                "reaction.yaml requires only UniReaction and BiReaction sections"
            )
        uni = {
            name: UniReaction(name=name, **value)
            for name, value in raw["UniReaction"].items()
        }
        bi = {}
        for name, value in raw["BiReaction"].items():
            # Normalize both allowed reactant orders once when loading. A
            # BiReaction now owns its direction; no separate BiAction is needed.
            for block_first in [False, True] if value["ordered"] else [False]:
                direction = "block_first" if block_first else "state_first"
                oriented_name = f"{name}_{direction}"
                bi[oriented_name] = BiReaction(
                    name=oriented_name,
                    forward=value["forward"],
                    reverse=value["reverse"],
                    block_types=tuple(value["block_types"]),
                    block_first=block_first,
                )
        return uni, bi

    def _compatible_block_types(self, site_type: int) -> list[str]:
        return self.blocks_by_attachment.get(site_type, [])

    def _build_signature(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for name in (
            "synthon.yaml",
            "reaction.yaml",
            "building_blocks.json",
            "bb_feature.npz",
        ):
            with (self.env_dir / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        for name in self.block_types:
            digest.update(name.encode())
            digest.update((self.env_dir / "blocks" / f"{name}.smi").read_bytes())
        return {
            "format": "rxnflow-environment",
            "block_counts": {name: len(library) for name, library in self.blocks.items()},
            "reaction_names": self.action_names,
            "content_sha256": digest.hexdigest(),
        }

    @staticmethod
    def initial_state() -> State:
        return State()

    @staticmethod
    def is_terminal(state: State) -> bool:
        return state.terminated

    @staticmethod
    def dummy_signature(smiles: str) -> tuple[int, ...]:
        mol = parse_molecule(smiles)
        return () if mol is None else typed_dummy_isotopes(mol)

    def get_action_space(self, state: State) -> ActionSpace:
        """Look up type/step eligibility; property masks follow subsampling."""
        if state.terminated:
            return []
        if state.mol is None:
            return self.action_space[None]
        if state.reaction_count >= self.max_reactions:
            return []
        signature = typed_dummy_isotopes(state.mol)
        if len(signature) != 1 or signature[0] not in self.synthon_types:
            return []
        last_step = state.reaction_count + 1 == self.max_reactions
        may_terminate = state.reaction_count + 1 >= self.min_reactions
        return self._step_action_space[signature[0], last_step, may_terminate]

    def block_mask(
        self,
        state_properties: torch.Tensor,
        block_type: str,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HSX budget rule on selected rows (or the full library for inspection).

        Use raw property units instead of division by the bound, so zero and
        negative upper bounds work too. These sums estimate product properties;
        they are not exact product descriptors. Enamine dummies have zero mass:
        do not copy HSX's Synple At subtraction or +29 MW linker correction.
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
        mask = (
            heavy_atoms + state_properties[..., PROPERTY_NAMES.index("heavy_atoms"), None]
            <= self.max_atoms
        )
        for index, limit in self.property_limits.items():
            estimate = properties[:, index] + state_properties[..., index, None]
            # HSX main uses normalized budget < 1.01 for positive bounds.
            # Compare raw units for zero/negative bounds, where division is not
            # meaningful. Discrete zero bounds remain exact; capacity is strict.
            if limit == 0:
                mask &= estimate <= 0
            else:
                mask &= estimate < limit + abs(limit) * 0.01
        return mask

    def _reaction_product(
        self, current: Chem.Mol | None, action: Action
    ) -> Chem.Mol | None:
        """Execute the selected action, including its concrete block index."""
        if action.block_type is not None:
            library = self.blocks[action.block_type]
            if action.block_index is None or not 0 <= action.block_index < len(library):
                raise ValueError("block_index is out of range")
            block_smiles = library.smiles[action.block_index]
        elif action.block_index is not None:
            raise ValueError("unary actions do not take a block_index")

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
        spec = (action.action_type, action.reaction or "first_block", action.block_type)
        if spec not in self.get_action_space(state):
            raise ValueError(f"action is not available for the current state: {action}")
        product = self._product(state.mol, action)
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
            result["block_role"] = "brick" if library.is_brick else "linker"
            result["block_smiles"] = library.smiles[action.block_index]
            result["block_ids"] = identifiers
            result["building_blocks"] = [
                {"id": identifier, "smiles": self.sources[identifier]}
                for identifier in identifiers
            ]
        return result

    def backward_log_probability(
        self, state: State, action: Action, parent_smiles: str
    ) -> float | None:
        return self.retro_analyzer.log_probability(
            state.smiles,
            state.reaction_count,
            action,
            self.num_total_actions,
            parent_smiles,
        )
