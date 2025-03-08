import argparse
import multiprocessing
from pathlib import Path

from _a_refine import get_clean_smiles
from tqdm import tqdm


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
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.sdf)"
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/enamine_catalog.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
