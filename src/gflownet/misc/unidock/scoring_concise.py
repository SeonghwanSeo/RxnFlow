from pathlib import Path
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol

from unidock_tools.modules.docking import run_unidock


def unidock_scores(
    rdmol_list: list[RDMol],
    pocket_file: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    search_mode: str = "balance",
    out_dir: Path | str | None = None,
) -> list[float]:
    docking_scores: list[float] = [0.0] * len(rdmol_list)
    with tempfile.TemporaryDirectory() as tempdir:
        root_dir = Path(tempdir)
        if out_dir is None:
            out_dir = root_dir / "docking"
            out_dir.mkdir()
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        etkdg_dir = root_dir / "etkdg"
        etkdg_dir.mkdir()

        sdf_list = []
        for i, mol in enumerate(rdmol_list):
            sdf_path = save_to_sdf(mol, i, etkdg_dir)
            if sdf_path is not None:
                sdf_list.append(Path(sdf_path))

        run_unidock(
            Path(pocket_file),
            sdf_list,
            Path(out_dir),
            center[0],
            center[1],
            center[2],
            num_modes=1,
            search_mode=search_mode,
            seed=seed,
        )
        for docked_sdf_file in out_dir.iterdir():
            idx = int(docked_sdf_file.stem.split("_")[0])
            docking_scores[idx] = parse_docked_file(docked_sdf_file)
    return docking_scores


def save_to_sdf(mol: RDMol, index: int, folder: Path | str, seed: int = 1) -> str | None:
    try:
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
    except:
        return None


def parse_docked_file(sdf_file_name) -> float:
    docking_score = 0
    with open(sdf_file_name) as f:
        for ln in f.readlines():
            if ln.startswith("ENERGY="):
                docking_score = float(ln.split()[1])
                break
    return min(0, docking_score)
