from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from rdkit.Chem import Mol as RDMol


class Reaction:
    def __init__(self, template: str):
        self.template: str = template
        self._rxn: ChemicalReaction = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(self._rxn)
        self.num_reactants: int = self._rxn.GetNumReactantTemplates()
        self.num_products: int = self._rxn.GetNumProductTemplates()

        self.reactant_pattern: list[RDMol] = []
        for i in range(self.num_reactants):
            self.reactant_pattern.append(self._rxn.GetReactantTemplate(i))

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if order is None:
            return self._rxn.IsMoleculeReactant(mol)
        else:
            return mol.HasSubstructMatch(self.reactant_pattern[order])

    def __call__(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: RDMol
                reactants
            strict: bool
                if strict, products should be yield.

        Returns:
            products: list[tuple[RDMol, ...]]
                The products of the reaction.
        """
        return self.forward(*reactants, strict=strict)

    def forward(self, *reactants: RDMol, strict: bool) -> list[tuple[RDMol, ...]]:
        """Perform in-silico reactions"""
        assert (
            len(reactants) == self.num_reactants
        ), f"number of inputs should be same to the number of reactants ({len(reactants)} vs {self.num_reactants})"
        ps: list[list[RDMol]] = self._rxn.RunReactants(reactants, 5)

        refine_ps: list[tuple[RDMol, ...]] = []
        for p in ps:
            if not len(p) == self.num_products:
                continue
            _ps = []
            for mol in map(_refine_mol, p):
                if mol is None:
                    break
                _ps.append(mol)
            if len(_ps) == self.num_products:
                refine_ps.append(tuple(_ps))
        del ps
        if strict:
            assert len(refine_ps) > 0, "Reaction did not yield any products."

        # remove redundant products
        unique_ps = []
        _storage = set()
        for p in refine_ps:
            key = tuple(Chem.MolToSmiles(mol) for mol in p)
            if key not in _storage:
                _storage.add(key)
                unique_ps.append(p)
        return unique_ps


class BiRxnReaction(Reaction):
    def __init__(self, template: str, is_block_first: bool):
        super().__init__(template)
        self.block_order: int = 0 if is_block_first else 1
        assert self.num_reactants == 2
        assert self.num_products == 1

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if order is not None:
            if self.block_order == 0:
                order = 1 - order
        return super().is_reactant(mol, order)

    def forward(self, *reactants: RDMol, strict: bool) -> list[tuple[RDMol, ...]]:
        if self.block_order == 0:
            reactants = tuple(reversed(reactants))
        return super().forward(*reactants, strict=strict)


class BckBiRxnReaction(Reaction):
    def __init__(self, template: str, is_block_first: bool):
        super().__init__(template)
        self.block_order: int = 0 if is_block_first else 1
        assert self.num_reactants == 1
        assert self.num_products == 2

    def forward(self, *reactants: RDMol, strict: bool) -> list[tuple[RDMol, ...]]:
        ps = super().forward(*reactants, strict=strict)
        if self.block_order == 0:
            ps = [(p[1], p[0]) for p in ps]
        return ps


def _refine_mol(mol: RDMol) -> RDMol | None:
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        mol = Chem.MolFromSmiles(smi, replacements={"[C]": "C", "[N]": "N", "[CH]": "C"})
    except Exception:
        return None
    return mol
