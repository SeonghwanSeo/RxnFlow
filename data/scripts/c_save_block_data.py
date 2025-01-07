from pathlib import Path
import argparse

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import multiprocessing

from rdkit import Chem
from rxnflow.envs.building_block import get_block_features


def run(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp, desc = get_block_features(mol)
    return fp, desc


def main(env_dir: str | Path, num_cpus: int):
    env_dir = Path(env_dir)
    block_smi_dir = env_dir / "blocks/"
    save_block_data_path = env_dir / "bb_feature.pt"

    data: dict[str, tuple[Tensor, Tensor]] = {}

    for smi_file in block_smi_dir.iterdir():
        with smi_file.open() as f:
            lines = f.readlines()
        smi_list = [ln.split()[0] for ln in lines]
        desc_list = []
        fp_list = []
        for idx in tqdm(range(0, len(smi_list), 10000)):
            chunk = smi_list[idx : idx + 10000]
            with multiprocessing.Pool(num_cpus) as pool:
                results = pool.map(run, chunk)
            for fp, desc in results:
                fp_list.append(fp)
                desc_list.append(desc)
        block_descs = torch.from_numpy(np.stack(desc_list, 0))
        block_fps = torch.from_numpy(np.stack(fp_list, 0))
        data[smi_file.stem] = (block_descs, block_fps)
    torch.save(data, save_block_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "--env_dir",
        type=str,
        help="Path to input synple building block directory",
        default="./envs/real/",
    )

    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    main(args.env_dir, args.cpu)
