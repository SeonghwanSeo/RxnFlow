import os

import numpy as np
import torch_geometric.data as gd
from rdkit import Chem
from torch import Tensor


class PocketDB:
    def __init__(self, pocket_graph_dict: dict[str, dict[str, Tensor]]):
        self.keys: list[str] = sorted(list(pocket_graph_dict.keys()))
        self.graphs: list[gd.Data] = [gd.Data(**pocket_graph_dict[key]) for key in self.keys]

        self.batch_idx: list[int]
        self.batch_keys: list[str]
        self.batch_g: gd.Batch

    def __len__(self):
        return len(self.keys)

    def set_batch_idcs(self, indices: list[int]):
        self.batch_idcs = indices
        self.batch_keys = [self.keys[i] for i in indices]
        self.batch_g = gd.Batch.from_data_list([self.graphs[i] for i in indices])

    def pocket_idx_to_batch_idx(self, indices: list[int]):
        return [self.batch_idcs.index(idx) for idx in indices]


def get_mol_center(ligand_path: str) -> tuple[float, float, float]:
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
