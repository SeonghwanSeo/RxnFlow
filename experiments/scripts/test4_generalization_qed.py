import time
from pathlib import Path
import torch
import random
import numpy as np
from gflownet.config import Config, init_empty
from gflownet.tasks.analysis_qed import QEDSynthesisSampler
import pickle

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    ckpt_dir = Path("./release-ckpt/generalization-qed")
    data_dir = Path("./data/envs/generalization")
    seen_dir = data_dir / "seen"
    unseen_dir = data_dir / "unseen"
    all_dir = data_dir / "all"

    save_dir = Path("./analysis/result/exp4/")
    save_dir.mkdir(parents=True)

    for beta in [1, 10, 20, 30]:
        print(beta)
        checkpoint_path = ckpt_dir / f"beta-{beta}/model_state.pt"
        for label, env_dir in zip(["seen", "unseen", "all"], [seen_dir, unseen_dir, all_dir], strict=True):
            print(label)
            config = init_empty(Config())
            config.env_dir = str(env_dir)
            config.algo.global_batch_size = 200
            config.algo.action_sampling.num_sampling_add_first_reactant = 10_000
            config.algo.action_sampling.max_sampling_reactbi = 10_000
            if label == "all":
                config.algo.action_sampling.sampling_ratio_reactbi = 10_000 / 1_000_000
            else:
                config.algo.action_sampling.sampling_ratio_reactbi = 10_000 / 500_000

            sampler = QEDSynthesisSampler(config, checkpoint_path, "cuda")

            # NOTE: Run
            st = time.time()
            res = sampler.sample(5000, calc_reward=True)
            runtime = time.time() - st
            rewards = [sample["info"]["reward"] for sample in res]
            mean = np.mean(rewards)
            std = np.std(rewards)
            print(f"mean: {mean:.3f}\tstd: {std:.3f}\ttime: {runtime:.1f}s")
            with open(save_dir / f"{label}_{beta}.pkl", "wb") as w:
                pickle.dump(res, w)
        print()
