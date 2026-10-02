"""RDKit forward and reverse reactions over synthon molecules."""

from __future__ import annotations

from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem.rdChemReactions import (
    ChemicalReaction,
    ReactionFromSmarts,
    ReactionToSmarts,
)


def _compile(smarts: str) -> ChemicalReaction:
    reaction = ReactionFromSmarts(smarts)
    if reaction is None:
        raise ValueError(f"invalid reaction SMARTS: {smarts}")
    reaction.Initialize()
    return reaction


def _run(
    reaction: ChemicalReaction, reactants: tuple[Chem.Mol, ...]
) -> list[tuple[Chem.Mol, ...]]:
    assert len(reactants) == reaction.GetNumReactantTemplates()
    products: dict[tuple[str, ...], tuple[Chem.Mol, ...]] = {}
    # SMILES are only deduplication/order keys. Return sanitized molecules so
    # atom/property checks do not need a Mol -> SMILES -> Mol round trip.
    for product_set in reaction.RunReactants(reactants, 0):
        if len(product_set) != reaction.GetNumProductTemplates():
            continue
        molecules: list[Chem.Mol] = []
        keys: list[str] = []
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
                product = Chem.RemoveHs(product)
                key = Chem.MolToSmiles(product)
            except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                break
            molecules.append(product)
            keys.append(key)
        if len(molecules) == len(product_set):
            products.setdefault(tuple(keys), tuple(molecules))
    return [products[key] for key in sorted(products)]


class Reaction:
    """Compiled forward/reverse SMARTS with sanitized, deduplicated products."""

    def __init__(self, name: str, forward: str, reverse: str) -> None:
        self.name = name
        self.forward_template = forward
        self.reverse_template = reverse
        self.forward_reaction = _compile(forward)
        self.reverse_reaction = _compile(reverse)

    def run_forward(self, *reactants: Chem.Mol) -> Chem.Mol | None:
        """Return the unique connected product, or None for an infeasible match.

        Incoming block orientation fixes the attachment site. Symmetry-related
        matches are deduplicated by _run; distinct products indicate a template
        ambiguity and must not be resolved by silently picking the first one.
        """
        products = [
            product_set[0]
            for product_set in _run(self.forward_reaction, tuple(reactants))
            if len(product_set) == 1 and len(Chem.GetMolFrags(product_set[0])) == 1
        ]
        if len(products) > 1:
            raise ValueError(
                f"reaction {self.name} produced {len(products)} distinct forward products"
            )
        return products[0] if products else None

    def run_reverse(self, product: Chem.Mol) -> list[tuple[Chem.Mol, ...]]:
        # _run already enumerates and deduplicates every RDKit match. Truncating
        # here can discard the only decomposition whose block is in the catalog.
        return _run(self.reverse_reaction, (product,))


class UniReaction(Reaction):
    """Transform one typed handle; output_type=None consumes it and terminates."""

    def __init__(
        self,
        name: str,
        forward: str,
        reverse: str,
        input_type: int,
        output_type: int | None,
    ) -> None:
        super().__init__(name, forward, reverse)
        self.input_type = input_type
        self.output_type = output_type
        if (
            self.forward_reaction.GetNumReactantTemplates() != 1
            or self.forward_reaction.GetNumProductTemplates() != 1
            or self.reverse_reaction.GetNumReactantTemplates() != 1
            or self.reverse_reaction.GetNumProductTemplates() != 1
        ):
            raise ValueError(f"invalid unary reaction shape: {self.name}")
        # Unary transformations must act on the marked handle, not an unrelated
        # ordinary functional group elsewhere in the molecule.
        for pattern, expected in (
            (self.forward_reaction.GetReactantTemplate(0), self.input_type),
            (self.forward_reaction.GetProductTemplate(0), self.output_type),
            (self.reverse_reaction.GetReactantTemplate(0), self.output_type),
            (self.reverse_reaction.GetProductTemplate(0), self.input_type),
        ):
            sites = [
                atom.GetIsotope()
                for atom in pattern.GetAtoms()
                if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0
            ]
            if sites != ([] if expected is None else [expected]):
                raise ValueError(
                    f"unary SMARTS site disagrees with its type: {self.name}"
                )


class BiReaction(Reaction):
    """A bimolecular template oriented as (state, incoming building block).

    The incoming block's unique isotope-0 dummy fixes its attachment site.
    Its chemical type is carried by the library key and ``block_type``. The
    reverse template recreates that marker so catalog lookup preserves direction.
    """

    def __init__(
        self,
        name: str,
        forward: str,
        reverse: str,
        block_types: tuple[int, int],
        block_first: bool = False,
    ) -> None:
        super().__init__(name, forward, reverse)
        self.block_types = block_types
        self.block_first = block_first
        if (
            self.forward_reaction.GetNumReactantTemplates() != 2
            or self.forward_reaction.GetNumProductTemplates() != 1
            or self.reverse_reaction.GetNumReactantTemplates() != 1
            or self.reverse_reaction.GetNumProductTemplates() != 2
        ):
            raise ValueError(f"invalid bimolecular reaction shape: {self.name}")
        if len(self.block_types) != 2 or any(value <= 0 for value in self.block_types):
            raise ValueError(f"invalid block types for reaction {self.name}")

        # Reorder both directions into (state, block), then change the block's
        # attachment query to isotope 0 while retaining the state's typed query.
        state_position, block_position = (1, 0) if self.block_first else (0, 1)
        forward = ChemicalReaction()
        reverse = ChemicalReaction()
        for position in (state_position, block_position):
            reactant = Chem.Mol(self.forward_reaction.GetReactantTemplate(position))
            product = Chem.Mol(self.reverse_reaction.GetProductTemplate(position))
            if position == block_position:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0:
                        # [0#0] matches only the attachment marker, not a typed
                        # dummy at the other end (nor an ordinary element).
                        atom.SetQuery(Chem.AtomFromSmarts("[0#0]"))
                for atom in product.GetAtoms():
                    if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0:
                        atom.SetQuery(Chem.AtomFromSmarts("[0#0]"))
                        atom.SetIsotope(0)
            forward.AddReactantTemplate(reactant)
            reverse.AddProductTemplate(product)
        forward.AddProductTemplate(self.forward_reaction.GetProductTemplate(0))
        reverse.AddReactantTemplate(self.reverse_reaction.GetReactantTemplate(0))
        # Compile independent templates after editing their queries/order.
        self.forward_reaction = _compile(ReactionToSmarts(forward))
        self.reverse_reaction = _compile(ReactionToSmarts(reverse))

    @property
    def state_type(self) -> int:
        return self.block_types[int(self.block_first)]

    @property
    def block_type(self) -> int:
        return self.block_types[int(not self.block_first)]


def load_reactions(path: Path) -> tuple[dict[str, UniReaction], dict[str, BiReaction]]:
    """Compile the named unary reactions and permitted binary orientations."""
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
        # Compile each permitted direction with the state first and incoming
        # block second, so runtime execution always uses the same argument order.
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
