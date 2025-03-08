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


def parse_temperature(temperature_param: str) -> tuple[str, list[float]]:
    temperature_info = temperature_param.split("-")
    sample_dist = temperature_info[0]
    dist_params = list(map(float, temperature_info[1:]))
    assert sample_dist in ("constant", "uniform", "loguniform", "gamma", "beta")
    if sample_dist == "constant":
        assert len(dist_params) == 1, "constant temperature requires only one parameter; e.g. constant-32"
    else:
        assert len(dist_params) == 2, f"{sample_dist} temperature requires two parameters."
    return sample_dist, dist_params
