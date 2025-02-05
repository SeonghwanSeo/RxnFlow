from pathlib import Path
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol

from unidock_tools.application.proteinprep import pdb2pdbqt
from unidock_tools.application.unidock_pipeline import UniDock


def unidock_scores(
    rdmol_list: list[RDMol],
    protein_file: str | Path,
    center: tuple[float, float, float],
    out_path: Path | str,
    size: float | tuple[float, float, float] = 22.5,
    seed: int = 1,
    search_mode: str = "balance",
    ff_optimization: str | None = None,
) -> list[float]:
    assert search_mode in ["fast", "balance", "detail"]
    assert ff_optimization in [None, "UFF", "MMFF"]

    if isinstance(size, int | float):
        size = (size, size, size)
    docking_scores: list[float] = [0.0] * len(rdmol_list)

    protein_file = Path(protein_file)
    protein_pdbqt_file = protein_file.parent / (protein_file.stem + ".pdbqt")
    if not protein_pdbqt_file.exists():
        pdb2pdbqt(protein_file, protein_pdbqt_file)

    with tempfile.TemporaryDirectory() as tempdir:
        root_dir = Path(tempdir)
        etkdg_dir = root_dir / "etkdg"
        etkdg_dir.mkdir(parents=True)
        unidock_dir = root_dir / "docking"
        unidock_dir.mkdir()

        # NOTE: run etkdg
        sdf_list = []
        for i, mol in enumerate(rdmol_list):
            sdf_path = run_etkdg(mol, i, etkdg_dir, seed, ff_optimization)
            if sdf_path is not None:
                sdf_list.append(Path(sdf_path))

        if len(sdf_list) > 0:
            cx, cy, cz = center
            sx, sy, sz = size
            # NOTE: run docking
            runner = UniDock(protein_pdbqt_file, sdf_list, cx, cy, cz, sx, sy, sz, workdir=root_dir)
            runner.docking(unidock_dir, search_mode=search_mode, num_modes=1, seed=seed)

        # NOTE: save
        with open(out_path, "w") as writer:
            for idx in range(len(rdmol_list)):
                score = 0
                docked_sdf_file = unidock_dir / f"{idx}.sdf"
                if docked_sdf_file.exists():
                    with open(docked_sdf_file) as f:
                        lines = f.readlines()
                    writer.writelines(lines)
                    for i, ln in enumerate(lines):
                        if ln.startswith(">  <docking_score>"):
                            score = float(lines[i + 1])
                            break
                docking_scores[idx] = min(score, 0)

    return docking_scores


def run_etkdg(mol: RDMol, index: int, folder: Path | str, seed: int = 1, ff_opt: str | None = None) -> str | None:
    assert ff_opt in [None, "UFF", "MMFF"]
    try:
        sdf_path = f"{folder}/{index}.sdf"
        mol = Chem.Mol(mol)
        mol = Chem.AddHs(mol)
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        AllChem.EmbedMolecule(mol, param)
        if ff_opt == "UFF":
            flag = AllChem.UFFOptimizeMolecule(mol)
            print(flag)
            if flag != 0:
                return None
        elif ff_opt == "MMFF":
            flag = AllChem.MMFFOptimizeMolecule(mol)
            if flag != 0:
                return None

        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            return None
        with Chem.SDWriter(sdf_path) as w:
            w.write(mol)
        return sdf_path
    except Exception as e:
        return None
