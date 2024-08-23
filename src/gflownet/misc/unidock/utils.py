import os
from pathlib import Path
import tempfile

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol


DEBUG = False


def unidock_scores(
    rdmol_list: list[RDMol],
    pocket_file: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    search_mode: str = "balance",
) -> list[float]:
    docking_scores: list[float] = [0.0] * len(rdmol_list)
    with tempfile.TemporaryDirectory() as tempdir:
        root_dir = Path(tempdir)
        out_dir = root_dir / "docking"
        out_dir.mkdir()
        etkdg_dir = root_dir / "etkdg"
        etkdg_dir.mkdir()
        index_path = root_dir / "index.txt"
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(save_to_sdf, mol, i, etkdg_dir, seed) for i, mol in enumerate(rdmol_list)]
            with index_path.open("w") as w:
                for feature in concurrent.futures.as_completed(futures):
                    sdf_path = feature.result()
                    if sdf_path is not None:
                        w.write(sdf_path + "\n")
        run_unidock(pocket_file, center, index_path, out_dir, seed, search_mode)
        for docked_sdf_file in out_dir.iterdir():
            idx = int(docked_sdf_file.stem)
            docking_scores[idx] = parse_docked_file(docked_sdf_file)
    return docking_scores


def run_unidock(
    pocket_file: str | Path,
    center: tuple[float, float, float],
    index_path: str | Path,
    save_dir: str | Path,
    seed: int = 1,
    search_mode: str = "balance",
):
    cx, cy, cz = center
    with tempfile.TemporaryDirectory() as tempdir:
        docking_cmd = f"unidocktools unidock_pipeline -r {pocket_file} -i {index_path} -sd {save_dir} -cx {cx:.2f} -cy {cy:.2f} -cz {cz:.2f} --seed {seed} -nm 1 --search_mode {search_mode} -wd {tempdir}"
        if not DEBUG:
            docking_cmd += ">/dev/null 2>/dev/null"
        os.system(docking_cmd)


def save_to_sdf(mol: RDMol, index: int, folder: Path | str, seed: int = 1) -> str | None:
    sdf_path = f"{folder}/{index}.sdf"
    mol = Chem.Mol(mol)
    mol = Chem.AddHs(mol)
    param = AllChem.srETKDGv3()
    param.randomSeed = seed
    AllChem.EmbedMolecule(mol, param)
    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return None
    with Chem.SDWriter(sdf_path) as w:
        w.write(mol)
    return sdf_path


def parse_docked_file(sdf_file_name) -> float:
    docking_score = 0
    with open(sdf_file_name) as f:
        flag = False
        for ln in f.readlines():
            if ln.startswith(">"):
                flag = True
            elif flag:
                docking_score = float(ln)
                break
    return min(0, docking_score)
