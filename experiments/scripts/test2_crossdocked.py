from pathlib import Path
import time
import torch
import numpy as np
import random
from tqdm import tqdm

from gflownet.config import Config, init_empty
from gflownet.tasks.sbdd_synthesis import SBDDSampler
from _exp2_constant import TEST_POCKET_DIR, TEST_POCKET_CENTER_INFO


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    env_root_dir = "./data/envs/enamine_all/"
    ckpt_path = Path("./release-ckpt/zero_shot_tacogfn_reward/model_state.pt")
    save_path = Path("./analysis/result/exp2/")
    save_path.mkdir(exist_ok=True, parents=True)
    config = init_empty(Config())
    config.env_dir = env_root_dir
    config.algo.global_batch_size = 100
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [32, 64]

    # NOTE: Run
    runtime = []
    sampler = SBDDSampler(config, ckpt_path, "cuda")
    for pocket_file in tqdm(sorted(list(Path(TEST_POCKET_DIR).iterdir()))):
        key = pocket_file.stem
        center = TEST_POCKET_CENTER_INFO[key]

        st = time.time()
        res = sampler.sample_against_pocket(pocket_file, center, 100, calc_reward=False)
        runtime.append(time.time() - st)

        with open(save_path / f"{key}.csv", "w") as w:
            w.write(",SMILES\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                w.write(f"{idx},{smiles}\n")
    print("avg time", np.mean(runtime))
