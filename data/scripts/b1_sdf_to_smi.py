from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing

from rdkit import Chem
from rdkit.Chem import BondType

ATOMS: list[str] = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def run(args):
    smiles, id = args
    mol = Chem.MolFromSmiles(smiles, replacements={"[2H]": "[H]"})

    # NOTE: Filtering Molecules with its structure
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    smi = Chem.MolToSmiles(mol)
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    fail = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ATOMS:
            fail = True
            break
    if fail:
        return None

    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDS:
            fail = True
            break
    if fail:
        return None
    return smi, id


def main(block_path: str, save_block_path: str, num_cpus: int):
    block_file = Path(block_path)
    assert block_file.suffix == ".sdf"

    print("Read SDF Files")
    with block_file.open() as f:
        lines = f.readlines()
    smiles_list = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <smiles>")]
    ids = [lines[i].strip() for i in tqdm(range(1, len(lines))) if lines[i - 1].startswith(">  <id>")]

    assert len(smiles_list) == len(ids), "sdf file error, number of <smiles> and <id> should be matched"
    print("Including Mols:", len(smiles_list))

    print("Run Building Blocks...")
    with open(save_block_path, "w") as w:
        for idx in tqdm(range(0, len(smiles_list), 10000)):
            chunk = list(zip(smiles_list[idx : idx + 10000], ids[idx : idx + 10000], strict=True))
            with multiprocessing.Pool(num_cpus) as pool:
                results = pool.map(run, chunk)
            for res in results:
                if res is None:
                    continue
                smiles, id = res
                w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.sdf)"
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/Enamine_Building_Blocks.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
