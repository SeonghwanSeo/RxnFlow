from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction


class Reaction:
    def __init__(self, template=None):
        self.template = template
        self.rxn = self.__init_reaction()
        self.reverse_rxn = self.__init_reverse_template()
        self.num_reactants = self.rxn.GetNumReactantTemplates()
        # Extract reactants, agents, products
        reactants, agents, products = self.template.split(">")
        if self.num_reactants == 1:
            self.reactant_template = list((reactants,))
        else:
            self.reactant_template = list(reactants.split("."))
        self.product_template = products

    # @cached_property
    def reactants(self):
        return self.rxn.GetReactants()

    def __init_reaction(self) -> ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(self.template)
        ChemicalReaction.Initialize(rxn)
        return rxn

    def __init_reverse_template(self) -> ChemicalReaction:
        """Reverses a reaction template and returns an initialized, reversed reaction object."""
        rxn = AllChem.ChemicalReaction()
        for i in range(self.rxn.GetNumReactantTemplates()):
            rxn.AddProductTemplate(self.rxn.GetReactantTemplate(i))
        for i in range(self.rxn.GetNumProductTemplates()):
            rxn.AddReactantTemplate(self.rxn.GetProductTemplate(i))
        rxn.Initialize()
        return rxn

    def is_product(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        return self.rxn.IsMoleculeProduct(mol)

    def is_reactant(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        return self.rxn.IsMoleculeReactant(mol)

    def is_reactant_first(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is the first reactant for the reaction."""
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is the second reactant for the reaction."""
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reactants(self, reactants: tuple[Chem.Mol, ...], safe=True) -> Chem.Mol | None:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            reactants: A tuple of reactants to run the reaction on.
            keep_main: Whether to return the main product or all products. Default is True.

        Returns:
            The product of the reaction or `None` if the reaction is not possible.
        """
        if len(reactants) not in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        if safe:
            if self.num_reactants == 1:
                if len(reactants) != 1:
                    raise ValueError(reactants)
                if not self.is_reactant(reactants[0]):
                    return None
            elif self.num_reactants == 2:
                if len(reactants) != 2:
                    raise ValueError(reactants)
                if not (self.is_reactant_first(reactants[0]) and self.is_reactant_second(reactants[1])):
                    return None
            else:
                raise ValueError("Reaction is neither unimolecular nor bimolecular.")

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
            assert rxn.IsMoleculeReactant(product)
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
    atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    if len(atoms_to_remove) > 0:
        rw_mol = Chem.RWMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            rw_mol.ReplaceAtom(idx, Chem.Atom("H"))
        try:
            rw_mol.UpdatePropertyCache()
            mol = rw_mol.GetMol()
            assert mol is not None
        except Exception as e:
            return None
    smi = Chem.MolToSmiles(mol)
    if "[CH]" in smi:
        smi = smi.replace("[CH]", "C")
    return Chem.MolFromSmiles(smi)
