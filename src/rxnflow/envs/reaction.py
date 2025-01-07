from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from rdkit.Chem import Mol as RDMol


class Reaction:
    def __init__(self, template: str):
        self._rxn: ChemicalReaction = self.__init_reaction(template)
        self.num_reactants: int = self._rxn.GetNumReactantTemplates()

    def __init_reaction(self, template: str) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if order is None:
            return self._rxn.IsMoleculeReactant(mol)
        else:
            # return mol.HasSubstructMatch(self.reactant_pattern[order])
            return mol.HasSubstructMatch(self._rxn.GetReactantTemplate(order))

    def __call__(self, *reactants: RDMol) -> list[tuple[RDMol, ...]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: RDMol
                reactants

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """
        return self.forward(*reactants)

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: reactants
            strict: force that the products exist

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """

        # Run reaction
        assert len(reactants) == self.num_reactants
        ps: list[list[RDMol]] = self._rxn.RunReactants(tuple(reactants), 10)
        if strict and len(ps) == 0:
            raise ValueError("Reaction did not yield any products.")

        refine_ps: list[tuple[RDMol, ...]] = []
        for p in ps:
            _p = []
            for mol in p:
                try:
                    mol = _refine_molecule(mol)
                except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
                    continue
                _p.append(mol)
            if len(_p) == len(p):
                refine_ps.append(tuple(_p))
        return refine_ps


class UniRxnReaction(Reaction):
    def __init__(self, template: str):
        super().__init__(template)
        assert self.num_reactants == 1

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        assert len(reactants) == 1
        return super().forward(*reactants, strict=strict)


class BckUniRxnReaction(Reaction):
    def __init__(self, template: str):
        super().__init__(template)
        assert self.num_reactants == 1

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        assert len(reactants) == 1
        return super().forward(*reactants, strict=strict)


class BiRxnReaction(Reaction):
    def __init__(self, template: str, is_block_first: bool):
        super().__init__(template)
        assert self.num_reactants == 2
        self.block_order: int = int(is_block_first)

    def is_block(self, mol: RDMol) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        return self.is_reactant(mol, self.block_order)

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if order is not None:
            if self.block_order == 0:
                order = 1 - order
        return super().is_reactant(mol, order)

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        assert len(reactants) == 2
        if self.block_order == 0:
            reactants = tuple(reversed(reactants))
        return super().forward(*reactants, strict=strict)


class BckBiRxnReaction(Reaction):
    def __init__(self, template: str, is_block_first: bool):
        super().__init__(template)
        assert self.num_reactants == 1
        self.block_order: int = int(is_block_first)

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        assert len(reactants) == 1
        ps = super().forward(*reactants, strict=strict)
        assert all(len(v) == 2 for v in ps), "number of products should be 2"
        if self.block_order == 0:
            ps = [(p[1], p[0]) for p in ps]
        return ps


def _refine_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)
    if "[CH]" in smi:
        smi = smi.replace("[CH]", "C")
    return Chem.MolFromSmiles(smi)
