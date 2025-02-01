from rdkit import Chem
from rdkit.Chem import BondType, SaltRemover
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

ATOMS = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


REFINE_ACID = ReactionFromSmarts("[C:1](=[O:2])[O-:3]>>[C:1](=[O:2])[OH:3]")
REFINE_ACID.Initialize()
remover = SaltRemover.SaltRemover()


def get_clean_smiles(smiles: str):
    if "[2H]" in smiles or "[13C]" in smiles:
        return None

    # smi -> mol
    mol = Chem.MolFromSmiles(smiles, replacements={"[C]": "C", "[CH]": "C", "[CH2]": "C", "[N]": "N"})
    try:
        assert mol is not None
        mol = Chem.RemoveHs(mol)
        mol = remover.StripMol(mol)
        Chem.SanitizeMol(mol)
        while REFINE_ACID.IsMoleculeReactant(mol):
            mol = REFINE_ACID.RunReactants((mol,), maxProducts=1)[0][0]
            Chem.SanitizeMol(mol)
    except Exception:
        return None

    # refine smi
    smi = Chem.MolToSmiles(mol)
    if smi is None:
        return None
    if len(smi.strip()) == 0:
        return None

    fail = False
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom: Chem.Atom
        if atom.GetSymbol() not in ATOMS:
            fail = True
            break
        elif atom.GetIsotope() != 0:
            fail = True
            break
        if atom.GetFormalCharge() not in [-1, 0, 1]:
            fail = True
            break
        if atom.GetNumExplicitHs() not in [0, 1]:
            fail = True
            break
    if fail:
        return None

    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDS:
            fail = True
            break
    if fail:
        return None
    return smi
