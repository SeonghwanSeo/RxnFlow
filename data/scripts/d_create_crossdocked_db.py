from pathlib import Path

import torch
from tqdm import tqdm

from rxnflow.appl.pocket_conditional.pocket.data import generate_protein_data

ROOT_DIR = Path("./experiments/CrossDocked2020/")

# for dataset in ["test", "train"]:
for dataset in ["train"]:
    PROTEIN_DIR = ROOT_DIR / "protein" / dataset
    CENTER_PATH = ROOT_DIR / "center_info" / f"{dataset}.csv"
    KEY_PATH = ROOT_DIR / "keys" / f"{dataset}.csv"

    infos = []
    with open(CENTER_PATH) as f:
        for ln in f.readlines():
            key, x, y, z = ln.split(",")
            infos.append((key, (float(x), float(y), float(z))))
    cache_dict = {}
    for key, xyz in tqdm(infos):
        protein_path = PROTEIN_DIR / f"{key}.pdb"
        cache_dict[key] = generate_protein_data(protein_path, xyz)
    torch.save(cache_dict, ROOT_DIR / f"{dataset}_db.pt")
