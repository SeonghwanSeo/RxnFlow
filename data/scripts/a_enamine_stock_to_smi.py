import argparse
import multiprocessing
from pathlib import Path

from _a_refine import get_clean_smiles
from rdkit import Chem
from tqdm import tqdm


def main(block_path: str, save_block_path: str, num_cpus: int):
    block_file = Path(block_path)
    assert block_file.suffix == ".sdf"

    print("Read SDF Files")
    mols = list(Chem.SDMolSupplier(str(block_file)))
    mols = [mol for mol in mols if mol is not None]
    ids = [mol.GetProp("ID") for mol in mols]
    print("Including Mols:", len(mols))
    print("Run Building Blocks...")
    clean_smiles_list = []
    for idx in tqdm(range(0, len(mols), 10000)):
        chunk = [Chem.MolToSmiles(mol) for mol in mols[idx : idx + 10000]]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(get_clean_smiles, chunk)
        clean_smiles_list.extend(results)

    with open(save_block_path, "w") as w:
        for smiles, id in zip(clean_smiles_list, ids, strict=True):
            if smiles is not None:
                w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get clean building blocks")
    parser.add_argument(
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.sdf)", required=True
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/enamine_stock.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers", default=1)
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
