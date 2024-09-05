import argparse
import os
from pathlib import Path
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors

from tqdm import tqdm

ROOT_DIR = Path("./envs/")
SEED = 1
SEED_ANALYSIS1 = 2  # NOTE: Due to typo error, we created environment for analysis1 study with seed = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample building blocks")
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        help="Path to root (entire) environment directory",
        default="./envs/enamine_all",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    template_path = root_dir / "template.txt"
    block_path = root_dir / "building_block.smi"
    mask_path = root_dir / "bb_mask.npy"
    fp_path = root_dir / "bb_fp_2_1024.npy"
    desc_path = root_dir / "bb_desc.npy"

    with open(block_path) as f:
        block_lines = f.readlines()

    random.seed(SEED)
    full_indices = list(range(len(block_lines)))
    random.shuffle(full_indices)

    mask = np.load(mask_path)
    fp = np.load(fp_path)
    desc = np.load(desc_path)
    print("load root environment")

    def save_with_indices(idcs: list[int], save_dir: Path):
        if save_dir.exists():
            return
        save_dir.mkdir(parents=True)
        save_template_path = save_dir / "template.txt"
        save_block_path = save_dir / "building_block.smi"
        save_mask_path = save_dir / "bb_mask.npy"
        save_fp_path = save_dir / "bb_fp_2_1024.npy"
        save_desc_path = save_dir / "bb_desc.npy"

        os.system(f"cp {template_path} {save_template_path}")
        with save_block_path.open("w") as w:
            for idx in idcs:
                w.write(block_lines[idx])
        np.save(save_mask_path, mask[:, :, idcs])
        np.save(save_fp_path, fp[idcs])
        np.save(save_desc_path, desc[idcs])

    # NOTE: Exp1
    print("start for exp1 - rgfn(350), synflownet(6000)")
    key_list = [(350, "rgfn_350"), (6_000, "synflownet_6k")]
    for n, key in key_list:
        idcs = sorted(full_indices[:n])
        save_dir = ROOT_DIR / key
        save_with_indices(idcs, save_dir)
    print("finish")

    # NOTE: Analysis2
    print("start for save analysis2 - low tpsa blocks")
    save_dir = ROOT_DIR / "restricted_low_tpsa"
    if not save_dir.exists():
        low_idcs = []
        for i, ln in enumerate(tqdm(block_lines)):
            block_smi = ln.split()[0]
            tpsa = Descriptors.TPSA(Chem.MolFromSmiles(block_smi), includeSandP=True)
            if tpsa < 30:
                low_idcs.append(i)
        save_with_indices(low_idcs, save_dir)
    print("finish")

    # NOTE: Analysis3
    print("start for save analysis3 - ablation study")
    key_list = [(100, "100"), (1_000, "1k"), (10_000, "10k"), (100_000, "100k"), (1_000_000, "1M")]
    for n, key in key_list:
        idcs = sorted(full_indices[:n])
        save_dir = ROOT_DIR / "ablation" / f"subsampled_{key}"
        save_with_indices(idcs, save_dir)
    print("finish")

    # NOTE: Analysis1
    print("start for save analysis1 - generalization: seen(500k), unseen(500k), all(1M)")
    random.seed(SEED_ANALYSIS1)
    full_indices = list(range(len(block_lines)))
    random.shuffle(full_indices)
    idcs1 = sorted(full_indices[0:500_000])
    idcs2 = sorted(full_indices[500_000:1_000_000])
    for key, idcs in [("unseen", idcs1), ("seen", idcs2), ("all", idcs1 + idcs2)]:
        save_dir = ROOT_DIR / "generalization" / key
        save_with_indices(idcs, save_dir)
    print("finish")
