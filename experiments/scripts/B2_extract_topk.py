import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")


def compute_diverse_top_k(
    smiles: list[str],
    rewards: list[float],
    k: int,
    thresh: float = 0.5,
) -> list[tuple[int, str, float]]:
    modes = [(i, smi, r) for i, (r, smi) in enumerate(zip(rewards, smiles, strict=True))]
    modes.sort(key=lambda m: m[2], reverse=True)
    top_modes = [modes[0]]

    prev_smis = {modes[0][1]}
    mode_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(modes[0][1]))]
    for i in range(1, len(modes)):
        smi = modes[i][1]
        if smi in prev_smis:
            continue
        prev_smis.add(smi)
        if thresh > 0:
            fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
            sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
            if max(sim) >= (1 - thresh):  # div = 1- sim
                continue
            mode_fps.append(fp)
            top_modes.append(modes[i])
        else:
            top_modes.append(modes[i])
        if len(top_modes) >= k:
            break
    return top_modes


ROOT_DIR = Path(sys.argv[1])
SAVE_DIR = Path("./result/benchmark/topk/") / ROOT_DIR.name
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "ADRB2",
    "ALDH1",
    "ESR_ago",
    "ESR_antago",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "TP53",
    "VDR",
]

avg_scores = []
for target in TARGETS:
    scores = []
    for trial in [0, 1, 2]:
        save_dir = SAVE_DIR / target
        save_dir.mkdir(exist_ok=True)
        save_filename = save_dir / f"seed-{trial}.csv"
        log_dir = ROOT_DIR / target / f"seed-{trial}"
        df = pd.read_csv(log_dir / "reward.csv")
        df = df[df.qed > 0.5]
        smiles = df.smiles.tolist()
        vinas = df.vina.tolist()
        qeds = df.qed.tolist()
        modes = compute_diverse_top_k(smiles, vinas, 100)
        smiles = [smiles[i] for i, *_ in modes]
        vinas = [vinas[i] for i, *_ in modes]
        qeds = [qeds[i] for i, *_ in modes]
        with save_filename.open("w") as w:
            w.write(",SMILES,Vina,QED\n")
            for i, (smi, vina, qed) in enumerate(zip(smiles, vinas, qeds, strict=True)):
                w.write(f"{i},{smi},{vina*-10},{qed}\n")

        df = pd.read_csv(save_filename)
        scores.append(df.Vina.to_numpy().mean())
        avg_scores.append(np.mean(scores))
    print(target, round(np.mean(scores), 2), round(np.std(scores), 2))

print("mean", np.mean(avg_scores))
print("median", np.median(avg_scores))
