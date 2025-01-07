from pathlib import Path
import argparse

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import multiprocessing

from rxnflow.envs.building_block import get_block_features


def get_block_data(env_dir: Path, num_cpus: int):
    block_smi_dir = env_dir / "blocks/"
    save_block_data_path = env_dir / "bb_feature.pt"

    data: dict[str, tuple[Tensor, Tensor]] = {}
    for smi_file in tqdm(list(block_smi_dir.iterdir())):
        with smi_file.open() as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        smi_list = [ln.split()[0] for ln in lines]
        fp_list = []
        desc_list = []
        for idx in tqdm(range(0, len(smi_list), 10000), leave=False):
            chunk = smi_list[idx : idx + 10000]
            with multiprocessing.Pool(num_cpus) as pool:
                results = pool.map(get_block_features, chunk)
            for fp, desc in results:
                fp_list.append(fp)
                desc_list.append(desc)
        block_descs = torch.from_numpy(np.stack(desc_list, 0))
        block_fps = torch.from_numpy(np.stack(fp_list, 0))
        data[smi_file.stem] = (block_descs, block_fps)
    torch.save(data, save_block_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-calculate building block features")
    parser.add_argument(
        "--env_dir",
        type=Path,
        help="Path of environment directory",
        default="./envs/real/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers")
    args = parser.parse_args()

    get_block_data(args.env_dir, args.cpu)
