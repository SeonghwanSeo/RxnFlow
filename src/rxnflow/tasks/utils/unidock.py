import os
import multiprocessing
from pathlib import Path
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import srETKDGv3, EmbedMolecule
from rdkit.Chem import Mol as RDMol, SDWriter
from openbabel import pybel
import warnings

from unidock_tools.application.proteinprep import pdb2pdbqt
from unidock_tools.application.unidock_pipeline import UniDock


class VinaReward:
    def __init__(
        self,
        protein_pdb_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
        size: tuple[float, float, float] = (22.5, 22.5, 22.5),
        search_mode: str = "fast",
        num_workers: int | None = None,
    ):
        self.protein_pdb_path: Path = Path(protein_pdb_path)
        if center is None:
            assert ref_ligand_path is not None, "reference ligand path is required"
            self.center = self.get_mol_center(ref_ligand_path)
        else:
            if ref_ligand_path is not None:
                warnings.warn("Both `center` and `ref_ligand_path` are given, so the reference ligand is ignored")
            self.center = center
        self.size = size
        self.search_mode = search_mode

        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        self.num_workers = num_workers

        self.history: dict[str, float] = {}

    def run_smiles(self, smiles_list: list[str], save_path: str | Path | None) -> list[float]:
        scores = [self.history.get(smi, 1) for smi in smiles_list]
        unique_indices = [i for i, v in enumerate(scores) if v > 0]
        if len(unique_indices) > 0:
            unique_objs = [Chem.MolFromSmiles(smiles_list[i]) for i in unique_indices]
            res = docking(
                unique_objs,
                self.protein_pdb_path,
                self.center,
                size=self.size,
                seed=1,
                search_mode=self.search_mode,
                num_workers=self.num_workers,
            )
            for j, (_, v) in zip(unique_indices, res, strict=True):
                scores[j] = min(v, 0.0)
                self.history[smiles_list[j]] = scores[j]
            if save_path is not None:
                with SDWriter(str(save_path)) as w:
                    for mol, _ in res:
                        if mol is not None:
                            w.write(mol)
        return scores

    def run_mols(self, mol_list: list[RDMol], save_path: str | Path | None) -> list[float]:
        smiles_list = [Chem.MolToSmiles(mol) for mol in mol_list]
        return self.run_smiles(smiles_list, save_path)

    @staticmethod
    def get_mol_center(ligand_path: str | Path) -> tuple[float, float, float]:
        format = Path(ligand_path).suffix[1:]
        pbmol: pybel.Molecule = next(pybel.readfile(format, str(ligand_path)))
        coords = [atom.coords for atom in pbmol.atoms]
        x, y, z = np.mean(coords, 0).tolist()
        return round(x, 2), round(y, 2), round(z, 2)


def docking(
    rdmols: list[RDMol],
    protein_pdb_path: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float | tuple[float, float, float] = 22.5,
    search_mode: str = "fast",
    num_workers: int = 4,
) -> list[tuple[None, float] | tuple[RDMol, float]]:

    # create pdbqt file
    protein_pdb_path = Path(protein_pdb_path)
    protein_pdbqt_path: Path = protein_pdb_path.parent / (protein_pdb_path.name + "qt")
    if not protein_pdbqt_path.exists():
        pdb2pdbqt(protein_pdb_path, protein_pdbqt_path)

    if isinstance(size, float | int):
        size = (size, size, size)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        sdf_list = []

        args = [(mol, out_dir / f"{i}.sdf") for i, mol in enumerate(rdmols)]
        with multiprocessing.Pool(num_workers) as pool:
            sdf_list = pool.map(run_etkdg_func, args)
        sdf_list = [file for file in sdf_list if file is not None]
        if len(sdf_list) > 0:
            runner = UniDock(
                protein_pdbqt_path,
                sdf_list,
                center[0],
                center[1],
                center[2],
                size[0],
                size[1],
                size[2],
                out_dir / "workdir",
            )
            runner.docking(
                out_dir / "savedir",
                num_modes=1,
                search_mode=search_mode,
                seed=seed,
            )

        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            try:
                docked_file = out_dir / "savedir" / f"{i}.sdf"
                docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))
            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
    return res


def run_etkdg_func(args: tuple[Chem.Mol, Path]) -> Path | None:
    # etkdg parameters
    param = srETKDGv3()
    param.randomSeed = 1
    param.timeout = 1  # prevent stucking

    mol, sdf_path = args
    if mol.GetNumAtoms() == 0 or mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol, param)
        assert mol.GetNumConformers() > 0
        mol = Chem.RemoveHs(mol)
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return None
    else:
        return sdf_path
