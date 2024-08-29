from pathlib import Path
import time
import torch
import numpy as np
import random

from gflownet.config import Config, init_empty
from gflownet.tasks.sbdd_synthesis import SBDDSampler
from _exp2_constant import TEST_POCKET_DIR, TEST_POCKET_CENTER_INFO


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    # checkpoint_path = Path("./storage/exp2/taco/qvina-CrossDocked2020/model_state_10000.pt")
    checkpoint_path = Path("./storage/exp2/taco-clip/qvina-CrossDocked2020/model_state_20000.pt")
    save_path = Path("./analysis/result/exp2-clip/ckpt-20000-32-64")
    save_path.mkdir(exist_ok=True)
    config = init_empty(Config())
    config.algo.global_batch_size = 100
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [32, 64]

    # NOTE: Run
    runtime = []
    sampler = SBDDSampler(config, checkpoint_path, "cuda")
    avg_qed = []
    for i, pocket_file in enumerate(Path(TEST_POCKET_DIR).iterdir()):
        set_seed(i)
        key = pocket_file.stem
        print(i, key)
        center = TEST_POCKET_CENTER_INFO[key]
        st = time.time()
        res = sampler.sample_against_pocket(pocket_file, center, 100, calc_reward=False)
        runtime.append(time.time() - st)
        _qed = []
        with open(save_path / f"{key}.csv", "w") as w:
            w.write(",SMILES,QED,PROXY\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                proxy = sample["info"]["reward_docking"]
                qed = sample["info"]["reward_qed"]
                w.write(f"{idx},{smiles},{qed:.4f},{proxy:.4f}\n")
                _qed.append(qed)
        avg_qed.append(np.mean(_qed))
        print("avg time", np.mean(runtime))
        print("avg qed", np.mean(avg_qed), np.median(avg_qed))


if __name__ == "__main__":
    main()
