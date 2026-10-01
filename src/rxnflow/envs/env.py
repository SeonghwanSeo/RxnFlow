"""Linear synthesis in synthon space: one growing intermediate, no Stop action."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from rxnflow.envs.chemistry.features import (
    PROPERTY_NAMES,
    heavy_atom_count,
    molecular_properties,
    parse_molecule,
)
from rxnflow.envs.chemistry.reaction import BiReaction, UniReaction
from rxnflow.envs.chemistry.synthon import load_synthon_specs, typed_dummy_isotopes
from rxnflow.types import ActionKind, MoleculeState, RxnAction

from .library import load_block_libraries


@dataclass(frozen=True)
class BiAction:
    name: str
    reaction: BiReaction
    state_type: int
    block_site_type: int
    block_first: bool


@dataclass(frozen=True)
class ActionGroup:
    kind: ActionKind
    name: str
    block_type: str | None = None


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
        self.bi_actions = self._expand_bi_actions()
        self.action_names = [
            "first_block",
            *sorted(self.uni_reactions),
            *sorted(self.bi_actions),
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
                for action in self.bi_actions.values()
                for name in self._compatible_block_types(action.block_site_type)
            ),
        )
        self.signature = self._build_signature()
        # Chemistry is independent of trajectory length; replay can reuse it.
        self._products = lru_cache(maxsize=8192)(self._reaction_products)

        from .retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSyntheticAnalyzer

        self.retro_analyzer = MultiRetroSyntheticAnalyzer(
            RetroSyntheticAnalyzer(self), retrosynthesis_workers
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
            bi[name] = BiReaction(
                name=name, **{**value, "block_types": tuple(value["block_types"])}
            )
        return uni, bi

    def _expand_bi_actions(self) -> dict[str, BiAction]:
        actions = {}
        for reaction in self.bi_reactions.values():
            left, right = reaction.block_types
            orientations = [(False, left, right)]
            if reaction.ordered:
                orientations = [(True, right, left), (False, left, right)]
            for block_first, state_type, block_type in orientations:
                name = f"{reaction.name}_{'b0' if block_first else 'b1'}"
                actions[name] = BiAction(
                    name, reaction, state_type, block_type, block_first
                )
        return actions

    def _compatible_block_types(self, site_type: int) -> list[str]:
        return [
            name for name in self.block_types if site_type in self.blocks[name].site_types
        ]

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
    def initial_state() -> MoleculeState:
        return MoleculeState()

    @staticmethod
    def is_terminal(state: MoleculeState) -> bool:
        return state.terminated

    @staticmethod
    def dummy_signature(smiles: str) -> tuple[int, ...]:
        mol = parse_molecule(smiles)
        return () if mol is None else typed_dummy_isotopes(mol)

    def available_groups(self, state: MoleculeState) -> list[ActionGroup]:
        if state.terminated:
            return []
        if not state.smiles:
            return [
                ActionGroup(ActionKind.FIRST_BLOCK, "first_block", name)
                for name in self.brick_types
            ]
        if state.reaction_count >= self.max_reactions:
            return []
        signature = self.dummy_signature(state.smiles)
        if len(signature) != 1:
            return []
        last_step = state.reaction_count + 1 == self.max_reactions
        may_terminate = state.reaction_count + 1 >= self.min_reactions
        groups = []
        for name, reaction in self.uni_reactions.items():
            if signature != (reaction.input_type,):
                continue
            terminal = reaction.output_type is None
            if (last_step and not terminal) or (terminal and not may_terminate):
                continue
            groups.append(ActionGroup(ActionKind.UNI_REACTION, name))
        for name, action in self.bi_actions.items():
            if signature != (action.state_type,):
                continue
            for block_type in self._compatible_block_types(action.block_site_type):
                terminal = self.blocks[block_type].is_brick
                if (last_step and not terminal) or (terminal and not may_terminate):
                    continue
                groups.append(ActionGroup(ActionKind.BI_REACTION, name, block_type))
        return groups

    def _reaction_products(
        self, smiles: str, group: ActionGroup, block_index: int | None
    ) -> tuple[str, ...]:
        """Execute chemistry and mask exact products, before policy selection."""
        if group.block_type is not None:
            library = self.blocks[group.block_type]
            if block_index is None or not 0 <= block_index < len(library):
                raise ValueError("block_index is out of range")
            block_smiles = library.smiles[block_index]
        elif block_index is not None:
            raise ValueError("unary actions do not take a block_index")

        if group.kind == ActionKind.FIRST_BLOCK:
            products = [block_smiles]
            expected = library.site_types
        else:
            current = parse_molecule(smiles)
            assert current is not None
            if group.kind == ActionKind.UNI_REACTION:
                reaction = self.uni_reactions[group.name]
                products = reaction.run_forward(current)
                expected = () if reaction.output_type is None else (reaction.output_type,)
            else:
                bi = self.bi_actions[group.name]
                block = parse_molecule(block_smiles)
                assert block is not None
                products = bi.reaction.run(current, block, bi.block_first)
                remaining = list(library.site_types)
                remaining.remove(bi.block_site_type)
                expected = tuple(remaining)
        feasible = []
        for product in products:
            mol = parse_molecule(product)
            assert mol is not None
            if typed_dummy_isotopes(mol) != expected or product == smiles:
                continue
            if heavy_atom_count(mol) > self.max_atoms:
                continue
            # These are hard limits on each resulting synthon (and on the final
            # dummy-free molecule). Reactant descriptor sums are not bounds:
            # templates may insert/delete atoms and descriptors are not additive.
            if self.property_limits:
                properties = molecular_properties(mol)
                if any(
                    properties[i] > limit for i, limit in self.property_limits.items()
                ):
                    continue
            feasible.append(product)
        return tuple(feasible)

    def outcomes(
        self, state: MoleculeState, group: ActionGroup, block_index: int | None = None
    ) -> list[RxnAction]:
        """Expand a compatible group/block into distinct, feasible site actions."""
        return [
            RxnAction(
                group.kind,
                product,
                reaction=None if group.kind == ActionKind.FIRST_BLOCK else group.name,
                block_type=group.block_type,
                block_index=block_index,
            )
            for product in self._products(state.smiles, group, block_index)
        ]

    def step(self, state: MoleculeState, action: RxnAction) -> MoleculeState:
        group = ActionGroup(
            action.kind, action.reaction or "first_block", action.block_type
        )
        if group not in self.available_groups(state) or action not in self.outcomes(
            state, group, action.block_index
        ):
            raise ValueError(f"action is not available for the current state: {action}")
        count = state.reaction_count + int(action.kind != ActionKind.FIRST_BLOCK)
        terminal = not self.dummy_signature(action.product_smiles)
        return MoleculeState(action.product_smiles, count, terminal)

    def action_to_dict(self, action: RxnAction) -> dict[str, object]:
        result: dict[str, object] = {
            "type": action.kind.name,
            "reaction": action.reaction,
            "product_smiles": action.product_smiles,
            "block_type": action.block_type,
            "block_index": action.block_index,
        }
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
        self, state: MoleculeState, action: RxnAction, parent_smiles: str
    ) -> float | None:
        return self.retro_analyzer.log_probability(
            state.smiles,
            state.reaction_count,
            action,
            self.num_total_actions,
            parent_smiles,
        )
