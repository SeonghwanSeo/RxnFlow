from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


class Reaction:
    def __init__(self, template: str):
        self.template: str = template
        self.rxn: ChemicalReaction = self.__init_forward()
        self.reverse_rxn: ChemicalReaction = self.__init_reverse()
        self.num_reactants: int = self.rxn.GetNumReactantTemplates()

    def reactants(self):
        return self.rxn.GetReactants()

    def __init_forward(self) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = ReactionFromSmarts(self.template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def __init_reverse(self) -> ChemicalReaction:
        """Reverses a reaction template and returns an initialized, reversed reaction object."""
        rxn = ChemicalReaction()
        for i in range(self.rxn.GetNumReactantTemplates()):
            rxn.AddProductTemplate(self.rxn.GetReactantTemplate(i))
        for i in range(self.rxn.GetNumProductTemplates()):
            rxn.AddReactantTemplate(self.rxn.GetProductTemplate(i))
        rxn.Initialize()
        return rxn

    def is_reactant(self, mol: Chem.Mol, order: int | None = None) -> bool:
        """Checks if a molecule is the first reactant for the reaction."""
        if order is None:
            return self.rxn.IsMoleculeReactant(mol)
        else:
            pattern = self.rxn.GetReactantTemplate(order)
            return mol.HasSubstructMatch(pattern)

    def is_product(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        return self.rxn.IsMoleculeProduct(mol)

    def run_reactants(self, reactants: tuple[Chem.Mol, ...], safe=True) -> Chem.Mol | None:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            reactants: A tuple of reactants to run the reaction on.
            keep_main: Whether to return the main product or all products. Default is True.

        Returns:
            The product of the reaction or `None` if the reaction is not possible.
        """
        if len(reactants) != self.num_reactants:
            raise ValueError(f"Can only run reactions with {self.num_reactants} reactants, not {len(reactants)}.")

        if safe:
            for i, mol in enumerate(reactants):
                if not self.is_reactant(mol, i):
                    return None

        # Run reaction
        ps = self.rxn.RunReactants(reactants)
        if len(ps) == 0:
            raise ValueError("Reaction did not yield any products.")
        p = ps[0][0]
        try:
            Chem.SanitizeMol(p)
            p = Chem.RemoveHs(p)
            return p
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException) as e:
            return None

    def run_reverse_reactants(self, product: Chem.Mol) -> list[Chem.Mol] | None:
        """Runs the reverse reaction on a product, to return the reactants.

        Args:
            product: A tuple of Chem.Mol object of the product (now reactant) to run the reverse reaction on.

        Returns:
            The product (reactant(s)) of the reaction or `None` if the reaction is not possible.
        """
        rxn = self.reverse_rxn
        try:
            rs_list = rxn.RunReactants((product,))
        except Exception:
            return None

        for rs in rs_list:
            if len(rs) != self.num_reactants:
                continue
            reactants = []
            for r in rs:
                if r is None:
                    break
                r = _refine_molecule(r)
                if r is None:
                    break
                reactants.append(r)
            if len(reactants) == self.num_reactants:
                return reactants
        return None


def _refine_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    smi = Chem.MolToSmiles(mol)
    if "[CH]" in smi:
        smi = smi.replace("[CH]", "C")
    return Chem.MolFromSmiles(smi)
