import os
import numpy as np
from rdkit import Chem


def get_center(ligand_path: str) -> tuple[float, float, float]:
    extension = os.path.splitext(ligand_path)[-1][1:]
    mol: Chem.Mol
    if extension == "sdf":
        mol = next(Chem.SDMolSupplier(ligand_path))
    elif extension == "mol2":
        mol = Chem.MolFromMol2File(ligand_path)
    elif extension == "pdb":
        mol = Chem.MolFromPDBFile(ligand_path)
    else:
        raise ValueError(f"{ligand_path} format should be `sdf`, `mol2`, or `pdb`")

    x, y, z = np.mean(mol.GetConformer().GetPositions(), axis=0).tolist()
    return round(x, 3), round(y, 3), round(z, 3)
