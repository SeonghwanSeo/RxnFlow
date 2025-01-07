from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing

from _utils import get_clean_smiles


def main(block_path: str, save_block_path: str, num_cpus: int):
    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI Files")
    with block_file.open() as f:
        lines = f.readlines()[1:]
    block_list: list[tuple[str, str]] = [tuple(ln.strip().split()) for ln in lines]

    clean_block_list = []
    for idx in tqdm(range(0, len(block_list), 10000)):
        chunk = block_list[idx : idx + 10000]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(get_clean_smiles, chunk)
        clean_block_list.extend(v for v in results if v is not None)

    with open(save_block_path, "w") as w:
        for smiles, id in clean_block_list:
            w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Building Blocks")
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=str,
        help="Path to input enamine building block file (.smi)",
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
