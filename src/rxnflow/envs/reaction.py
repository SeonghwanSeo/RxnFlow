from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts

from rdkit.Chem import Mol as RDMol


class Reaction:
    def __init__(self, template: str):
        self.template: str = template
        self._rxn_forward: ChemicalReaction = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(self._rxn_forward)
        self.num_reactants: int = self._rxn_forward.GetNumReactantTemplates()
        self.num_products: int = self._rxn_forward.GetNumProductTemplates()

        self.reactant_pattern: list[RDMol] = []
        for i in range(self.num_reactants):
            self.reactant_pattern.append(self._rxn_forward.GetReactantTemplate(i))

        # set reverse reaction
        self._rxn_reverse = ChemicalReaction()
        for i in range(self.num_reactants):
            self._rxn_reverse.AddProductTemplate(self._rxn_forward.GetReactantTemplate(i))
        for i in range(self.num_products):
            self._rxn_reverse.AddReactantTemplate(self._rxn_forward.GetProductTemplate(i))
        self._rxn_reverse.Initialize()

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Checks if a molecule is the reactant for the reaction."""
        if order is None:
            return self._rxn_forward.IsMoleculeReactant(mol)
        else:
            return mol.HasSubstructMatch(self.reactant_pattern[order])

    def is_product(self, mol: RDMol) -> bool:
        """Checks if a molecule is the product for the reaction."""
        return self._rxn_forward.IsMoleculeProduct(mol)

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        """Perform in-silico reactions"""
        assert (
            len(reactants) == self.num_reactants
        ), f"number of inputs should be same to the number of reactants ({len(reactants)} vs {self.num_reactants})"
        ps = _run_reaction(self._rxn_forward, reactants, self.num_reactants, self.num_products)
        if strict:
            assert len(ps) > 0, "ChemicalReaction did not yield any products."
        return ps

    def reverse(self, product: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        """Perform in-silico reactions"""
        rs = _run_reaction(self._rxn_reverse, (product,), self.num_products, self.num_reactants)
        if strict:
            assert len(rs) > 0, "ChemicalReaction did not yield any reactants."
        return rs


class UniReaction(Reaction):
    def __init__(self, template: str):
        super().__init__(template)
        assert self.num_reactants == 1
        assert self.num_products == 1


class BiReaction(Reaction):
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

    def forward(self, *reactants: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        if self.block_order == 0:
            reactants = tuple(reversed(reactants))
        return super().forward(*reactants, strict=strict)

    def reverse(self, product: RDMol, strict: bool = False) -> list[tuple[RDMol, ...]]:
        rs = super().reverse(product, strict=strict)
        if self.block_order == 0:
            rs = [(r[1], r[0]) for r in rs]
        return rs


def _run_reaction(
    reaction: ChemicalReaction,
    reactants: tuple[RDMol, ...],
    num_reactants: int,
    num_products: int,
) -> list[tuple[RDMol, ...]]:
    """Perform in-silico reactions"""
    assert len(reactants) == num_reactants
    ps: list[list[RDMol]] = reaction.RunReactants(reactants, 5)

    refine_ps: list[tuple[RDMol, ...]] = []
    for p in ps:
        if not len(p) == num_products:
            continue
        _ps = []
        for mol in p:
            try:
                mol = _refine_mol(mol)
                assert mol is not None
            except Exception as e:
                break
            _ps.append(mol)
        if len(_ps) == num_products:
            refine_ps.append(tuple(_ps))
    # remove redundant products
    unique_ps = []
    _storage = set()
    for p in refine_ps:
        key = tuple(Chem.MolToSmiles(mol) for mol in p)
        if key not in _storage:
            _storage.add(key)
            unique_ps.append(p)
    return unique_ps


def _refine_mol(mol: RDMol) -> RDMol | None:
    try:
        # mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, replacements={"[C]": "C", "[N]": "N", "[CH]": "C"})
    except Exception:
        return None
    return mol
