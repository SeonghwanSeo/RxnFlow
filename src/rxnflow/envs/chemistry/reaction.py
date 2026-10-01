"""Strict forward and reverse RDKit reaction definitions."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


def _compile(smarts: str) -> ChemicalReaction:
    reaction = ReactionFromSmarts(smarts)
    if reaction is None:
        raise ValueError(f"invalid reaction SMARTS: {smarts}")
    reaction.Initialize()
    return reaction


def _run(
    reaction: ChemicalReaction, reactants: tuple[Chem.Mol, ...]
) -> list[tuple[str, ...]]:
    assert len(reactants) == reaction.GetNumReactantTemplates()
    products: set[tuple[str, ...]] = set()
    # Enumerate all positional outcomes before canonical deduplication. A small
    # maxProducts can be exhausted by symmetry matches and hide valid sites.
    for product_set in reaction.RunReactants(reactants, 0):
        if len(product_set) != reaction.GetNumProductTemplates():
            continue
        values: list[str] = []
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(product))
                checked = Chem.MolFromSmiles(smiles)
            except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                break
            if checked is None:
                break
            values.append(Chem.MolToSmiles(checked))
        if len(values) == len(product_set):
            products.add(tuple(values))
    return sorted(products)


@dataclass(frozen=True)
class Reaction:
    name: str
    forward: str
    reverse: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "forward_reaction", _compile(self.forward))
        object.__setattr__(self, "reverse_reaction", _compile(self.reverse))

    def run_forward(self, *reactants: Chem.Mol) -> list[str]:
        products = _run(self.forward_reaction, tuple(reactants))
        return sorted(
            {value[0] for value in products if len(value) == 1 and "." not in value[0]}
        )

    def run_reverse(self, product: Chem.Mol, limit: int = 2) -> list[tuple[str, ...]]:
        return _run(self.reverse_reaction, (product,))[:limit]


@dataclass(frozen=True)
class UniReaction(Reaction):
    input_type: int
    output_type: int | None

    def __post_init__(self) -> None:
        super().__post_init__()
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


@dataclass(frozen=True)
class BiReaction(Reaction):
    block_types: tuple[int, int]
    ordered: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.forward_reaction.GetNumReactantTemplates() != 2
            or self.forward_reaction.GetNumProductTemplates() != 1
            or self.reverse_reaction.GetNumReactantTemplates() != 1
            or self.reverse_reaction.GetNumProductTemplates() != 2
        ):
            raise ValueError(f"invalid bimolecular reaction shape: {self.name}")
        if len(self.block_types) != 2 or any(value <= 0 for value in self.block_types):
            raise ValueError(f"invalid block types for reaction {self.name}")

    def run(self, state: Chem.Mol, block: Chem.Mol, block_first: bool) -> list[str]:
        reactants = (block, state) if block_first else (state, block)
        return self.run_forward(*reactants)

    def reverse_pairs(
        self, product: Chem.Mol, block_first: bool, limit: int = 2
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for products in self.run_reverse(product, limit):
            if len(products) != 2:
                continue
            block, state = products if block_first else (products[1], products[0])
            pairs.append((state, block))
        return pairs
