from pathlib import Path
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.modules.docking import run_unidock
from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed

        # NOTE: '*' -> 'C'
        rwmol = Chem.RWMol(mol)
        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() == "*":
                rwmol.ReplaceAtom(atom.GetIdx(), Chem.Atom("C"))
        mol = rwmol.GetMol()
        mol.UpdatePropertyCache()

        # NOTE: get etkdg structure
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, param)
        mol = Chem.RemoveHs(mol)
        assert mol.GetNumConformers() > 0
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return False
    else:
        return True


def run(
    rdmol: RDMol,
    pocket_file: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float = 20.0,
    search_mode: str = "fast",
):
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        ligand_file = out_dir / "ligand.sdf"
        flag = run_etkdg(rdmol, ligand_file)

        if flag is not None:
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
        try:
            docked_file = out_dir / "ligand_out.sdf"
            obj = list(Chem.SDMolSupplier(str(docked_file)))[0]
            assert obj is not None
            unidock_res = obj.GetProp("Uni-Dock RESULT")
            docking_score = float(unidock_res.split()[1])
        except:
            return None, 0.0
        return obj, docking_score


def run_batch(
    rdmols: list[RDMol],
    pocket_file: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float = 20.0,
    search_mode: str = "fast",
):
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        sdf_list = []
        for i, mol in enumerate(rdmols):
            ligand_file = out_dir / f"{i}.sdf"
            flag = run_etkdg(mol, ligand_file)
            if flag:
                sdf_list.append(ligand_file)
        if len(sdf_list) > 0:
            run_unidock(
                Path(pocket_file),
                sdf_list,
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

        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            try:
                docked_file = out_dir / f"{i}_out.sdf"
                docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                unidock_res = docked_rdmol.GetProp("Uni-Dock RESULT")
                docking_score = float(unidock_res.split()[1])
            except:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
        return res
