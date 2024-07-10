import torch
from gflownet.config import Config, init_empty
from gflownet.tasks.seh_synthesis import SEHSynthesisTrainer


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 100
    config.num_training_steps = 2000
    config.log_dir = "./logs/debug1/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/20240610_all/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 10_000
    config.algo.action_sampling.ratio_sampling_reactbi = 0.01
    config.algo.action_sampling.max_sampling_reactbi = 2_500
    config.algo.action_sampling.min_sampling_reactbi = 2_500

    trial = SEHSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
