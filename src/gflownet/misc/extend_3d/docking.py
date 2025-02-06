from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.modules.docking import run_unidock


def save_to_sdf(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> str | None:
    try:
        mol = Chem.Mol(mol)
        mol = Chem.AddHs(mol)
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        AllChem.EmbedMolecule(mol, param)
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            return None
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
        return str(sdf_path)
    except:
        return None


def run_docking(
    rdmol: RDMol,
    pocket_file: str | Path,
    center: tuple[float, float, float],
    out_dir: Path | str,
    seed: int = 1,
    size: float = 22.5,
    search_mode: str = "balance",
):
    out_dir = Path(out_dir)
    ligand_file = out_dir / "ligand.sdf"
    save_to_sdf(rdmol, ligand_file)

    run_unidock(
        Path(pocket_file),
        [ligand_file],
        out_dir,
        center[0],
        center[1],
        center[2],
        size,
        size,
        size,
        num_modes=1,
        search_mode=search_mode,
        seed=seed,
    )


def run_docking_list(
    rdmols: list[RDMol],
    pocket_file: str | Path,
    center: tuple[float, float, float],
    out_dir: Path | str,
    seed: int = 1,
    size: float = 22.5,
    search_mode: str = "balance",
):
    out_dir = Path(out_dir)
    etkdg_dir = out_dir / "etkdg"
    etkdg_dir.mkdir(exist_ok=True)
    docking_dir = out_dir / "docking"
    docking_dir.mkdir(exist_ok=True)
    sdf_list = []
    for i, mol in enumerate(rdmols):
        sdf_path = save_to_sdf(mol, etkdg_dir / f"{i}.sdf")
        if sdf_path is not None:
            sdf_list.append(Path(sdf_path))

    run_unidock(
        Path(pocket_file),
        sdf_list,
        docking_dir,
        center[0],
        center[1],
        center[2],
        size,
        size,
        size,
        num_modes=1,
        search_mode=search_mode,
        seed=seed,
    )
    mols = []
    for i in range(len(rdmols)):
        ligand_filename = docking_dir / f"{i}_out.sdf"
        if ligand_filename.exists():
            try:
                out = list(Chem.SDMolSupplier(str(ligand_filename)))
            except:
                mols.append(None)
            else:
                if len(out) > 0:
                    mols.append(out[0])
                else:
                    mols.append(None)
        else:
            mols.append(None)
    return mols
