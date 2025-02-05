import numpy as np

from rxnflow.config import Config, init_empty
from sampling_qed import QEDSampler

# NOTE: example setting
NUM_SAMPLES = 128
DEVICE = "cuda"


# TODO: add diversity

if __name__ == "__main__":
    # do not change config from training
    config = init_empty(Config())

    # construct sampler
    ckpt_path = "./logs/example-qed/model_state.pt"
    sampler = QEDSampler(config, ckpt_path, DEVICE)

    # change temperature for temperature-conditioned GFN
    # this is not allowed for non temperature-conditioned GFN (sample_dist='constant')
    # Only `loguniform` and `uniform` are compatible with each other."
    sampler.update_temperature("uniform", [0, 16])
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    qeds = [sample["info"]["reward"][0] for sample in res]
    print("avg qed:", np.mean(qeds))

    sampler.update_temperature("uniform", [60, 64])
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    qeds = [sample["info"]["reward"][0] for sample in res]
    print("avg qed:", np.mean(qeds))
