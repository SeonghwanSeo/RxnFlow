"""Strict RDKit reaction execution."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


class Reaction:
    def __init__(self, smarts: str):
        reaction = ReactionFromSmarts(smarts)
        if reaction is None:
            raise ValueError(f"invalid reaction SMARTS: {smarts}")
        reaction.Initialize()
        self.smarts = smarts
        self.reaction: ChemicalReaction = reaction

    @property
    def num_reactants(self) -> int:
        return self.reaction.GetNumReactantTemplates()

    def run(self, *reactants: Chem.Mol) -> str:
        assert len(reactants) == self.num_reactants
        raw_products = self.reaction.RunReactants(tuple(reactants), 10)
        products: set[str] = set()
        for product_set in raw_products:
            if len(product_set) != 1:
                continue
            product = product_set[0]
            try:
                Chem.SanitizeMol(product)
                product = Chem.RemoveHs(product)
                smiles = Chem.MolToSmiles(product).replace("[CH]", "C")
                checked = Chem.MolFromSmiles(smiles)
            except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                continue
            if checked is not None:
                products.add(Chem.MolToSmiles(checked))
        if len(products) != 1:
            reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
            raise ValueError(
                f"reaction must yield one unique single-molecule product; got {len(products)} for {reactant_smiles}"
            )
        return next(iter(products))
