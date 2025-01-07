from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing

from _a_refine import get_clean_smiles


def main(block_path: str, save_block_path: str, num_cpus: int):
    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI Files")
    with block_file.open() as f:
        lines = f.readlines()[1:]
    smiles_list = [ln.strip().split()[0] for ln in lines]
    ids = [ln.strip().split()[1] for ln in lines]
    print("Including Mols:", len(smiles_list))

    print("Run Building Blocks...")
    clean_smiles_list = []
    for idx in tqdm(range(0, len(smiles_list), 10000)):
        chunk = smiles_list[idx : idx + 10000]
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
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.smi)", required=True
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/enamine_blocks.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers", default=1)
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
