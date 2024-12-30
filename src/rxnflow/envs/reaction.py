from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from rdkit.Chem import Mol as RDMol


class Reaction:
    def __init__(self, template: str):
        self._rxn: ChemicalReaction = self.__init_reaction(template)
        self.num_reactants: int = self._rxn.GetNumReactantTemplates()
        # self.reactant_pattern: list[Chem.Mol] = []
        # for i in range(self.num_reactants):
        #     self.reactant_pattern.append(self._rxn.GetReactantTemplate(i))

    def __init_reaction(self, template: str) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def is_reactant(self, mol: RDMol, order: int) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        # return mol.HasSubstructMatch(self.reactant_pattern[order])
        return mol.HasSubstructMatch(self._rxn.GetReactantTemplate(order))

    def __call__(self, *reactants: RDMol) -> list[list[RDMol]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: RDMol
                reactants

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """
        return self.forward(*reactants)

    def forward(self, *reactants: RDMol) -> list[list[RDMol]]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            *reactants: reactants

        Returns:
            producs: list[list[RDMol]]
                The products of the reaction.
        """

        # Run reaction
        assert len(reactants) == self.num_reactants
        ps: list[list[RDMol]] = self._rxn.RunReactants(tuple(reactants), 10)
        if len(ps) == 0:
            raise ValueError("Reaction did not yield any products.")

        refine_ps: list[list[RDMol]] = []
        for p in ps:
            _p = []
            for mol in p:
                try:
                    mol = _refine_molecule(mol)
                except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
                    continue
                _p.append(mol)
            if len(_p) == len(p):
                refine_ps.append(_p)
        return refine_ps


def _refine_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)
    if "[CH]" in smi:
        smi = smi.replace("[CH]", "C")
    return Chem.MolFromSmiles(smi)
