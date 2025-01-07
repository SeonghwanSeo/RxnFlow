from rdkit import Chem
from rdkit.Chem import BondType

ATOMS = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def get_clean_smiles(smiles: str):
    if "[2H]" in smiles or "[13C]" in smiles:
        return None

    # smi -> mol
    mol = Chem.MolFromSmiles(smiles, replacements={"[C]": "C", "[CH]": "C", "[CH2]": "C", "[N]": "N"})
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # refine smi
    smi = Chem.MolToSmiles(mol)
    if smi is None:
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
