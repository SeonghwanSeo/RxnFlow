from pathlib import Path
import torch
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from gflownet.config import Config, init_empty
from gflownet.tasks.sbdd_synthesis import SBDDSampler

PROTEIN_PATH = "./data/experiments/KRAS-G12C/6oim_protein.pdb"
POCKET_CENTER = (1.872, -8.260, -1.361)


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    env_all_dir = "./data/envs/enamine_all/"
    env_low_tpsa_dir = "./data/envs/restricted_low_tpsa/"
    ckpt_path = Path("./release-ckpt/zero_shot_tacogfn_reward/model_state.pt")
    save_dir = Path("./analysis/result/exp3/")

    save_dir.mkdir(parents=True)
    for env_name in ["all", "low_tpsa"]:
        save_file = save_dir / f"{env_name}.csv"

        config = init_empty(Config())
        config.algo.global_batch_size = 100
        config.cond.temperature.sample_dist = "uniform"
        config.cond.temperature.dist_params = [32, 64]
        if env_name != "all":
            config.algo.action_sampling.sampling_ratio_reactbi = 0.1
            config.algo.action_sampling.num_sampling_add_first_reactant = 12_000
            config.algo.action_sampling.max_sampling_reactbi = 12_000
            config.env_dir = env_low_tpsa_dir
        else:
            config.env_dir = env_all_dir

        # NOTE: Run
        sampler = SBDDSampler(config, ckpt_path, "cuda")
        res = sampler.sample_against_pocket(PROTEIN_PATH, POCKET_CENTER, 500, calc_reward=False)
        with save_file.open("w") as w:
            w.write(",SMILES,QED,TPSA\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                qed = QED.qed(mol)
                tpsa = Descriptors.TPSA(mol)
                w.write(f"{idx},{smiles},{qed},{tpsa}\n")
        print(save_file)
