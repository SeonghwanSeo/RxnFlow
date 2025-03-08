import time

import numpy as np
from sampling_qed import QEDSampler

from rxnflow.config import Config, init_empty

# NOTE: example setting
SEED = 1
NUM_SAMPLES = 200
DEVICE = "cpu"


if __name__ == "__main__":
    # checkpoint path
    ckpt_path = "./logs/example/qed/model_state.pt"

    # change the parameter except for temperature
    config = init_empty(Config())
    config.algo.num_from_policy = 100  # batch size: 64 -> 100
    config.env_dir = "./data/envs/stock"  # if you want to use catalog, just remove this line

    # construct sampler
    sampler = QEDSampler(config, ckpt_path, DEVICE)

    # change temperature for temperature-conditioned GFN
    # this is not allowed for non temperature-conditioned GFN (sample_dist='constant')
    # Only `loguniform` and `uniform` are compatible with each other."
    st = time.time()
    sampler.update_temperature("uniform", [0, 16])
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    qeds = [sample["info"]["reward"][0] for sample in res]
    print(f"avg qed: {np.mean(qeds):.4f} ({time.time() - st:.3f} sec)")

    st = time.time()
    sampler.update_temperature("uniform", [32, 48])
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    qeds = [sample["info"]["reward"][0] for sample in res]
    print(f"avg qed: {np.mean(qeds):.4f} ({time.time() - st:.3f} sec)")

    st = time.time()
    sampler.update_temperature("uniform", [60, 64])
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    qeds = [sample["info"]["reward"][0] for sample in res]
    print(f"avg qed: {np.mean(qeds):.4f} ({time.time() - st:.3f} sec)")
