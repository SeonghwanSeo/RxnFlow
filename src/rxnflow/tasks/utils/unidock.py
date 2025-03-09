import multiprocessing
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem import SDWriter
from rdkit.Chem.rdDistGeom import EmbedMolecule, srETKDGv3
from unidock_tools.application.proteinprep import pdb2pdbqt
from unidock_tools.application.unidock_pipeline import UniDock


def get_mol_center(ligand_path: str | Path) -> tuple[float, float, float]:
    format = Path(ligand_path).suffix[1:]
    pbmol: pybel.Molecule = next(pybel.readfile(format, str(ligand_path)))
    coords = [atom.coords for atom in pbmol.atoms]
    x, y, z = np.mean(coords, 0).tolist()
    return round(x, 2), round(y, 2), round(z, 2)


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
            assert ref_ligand_path is not None, "One of center or reference ligand path is required"
            self.center = get_mol_center(ref_ligand_path)
        else:
            if ref_ligand_path is not None:
                warnings.warn(
                    "Both `center` and `ref_ligand_path` are given, so the reference ligand is ignored", stacklevel=2
                )
            self.center = center
        self.size = size
        self.search_mode = search_mode

        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        self.num_workers = num_workers

        self.history: dict[str, float] = {}

    def run_smiles(self, smiles_list: list[str], save_path: str | Path | None = None) -> list[float]:
        scores = [self.history.get(smi, 1) for smi in smiles_list]
        unique_indices = [i for i, v in enumerate(scores) if v > 0]
        if len(unique_indices) > 0:
            unique_smiles = [smiles_list[i] for i in unique_indices]
            res = docking(
                unique_smiles,
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

    def run_mols(self, mol_list: list[RDMol], save_path: str | Path | None = None) -> list[float]:
        smiles_list = [Chem.MolToSmiles(mol) for mol in mol_list]
        return self.run_smiles(smiles_list, save_path)


def docking(
    smiles_list: list[str],
    protein_pdb_path: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: tuple[float, float, float] = (22.5, 22.5, 22.5),
    search_mode: str = "fast",
    num_workers: int = 4,
) -> list[tuple[None, float] | tuple[RDMol, float]]:
    num_mols = len(smiles_list)

    # create pdbqt file
    protein_pdb_path = Path(protein_pdb_path)
    protein_pdbqt_path: Path = protein_pdb_path.parent / (protein_pdb_path.name + "qt")
    if not protein_pdbqt_path.exists():
        pdb2pdbqt(protein_pdb_path, protein_pdbqt_path)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        sdf_list = []

        # etkdg
        etkdg_dir = out_dir / "etkdg"
        etkdg_dir.mkdir(parents=True)
        args = [(smi, etkdg_dir / f"{i}.sdf") for i, smi in enumerate(smiles_list)]
        with multiprocessing.Pool(num_workers) as pool:
            sdf_list = pool.map(run_etkdg_func, args)
        sdf_list = [file for file in sdf_list if file is not None]

        # unidock
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
        for i in range(num_mols):
            try:
                docked_file = out_dir / "savedir" / f"{i}.sdf"
                docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))
            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
    return res


def run_etkdg_func(args: tuple[str, Path]) -> Path | None:
    # etkdg parameters
    param = srETKDGv3()
    param.randomSeed = 1
    param.timeout = 1  # prevent stucking

    smi, sdf_path = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol.GetNumAtoms() == 0 or mol is None:
            return None
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
