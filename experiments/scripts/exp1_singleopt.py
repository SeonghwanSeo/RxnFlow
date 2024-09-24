import os
import sys

import wandb
from omegaconf import OmegaConf
from _exp1_constant import TARGET_CENTER, TARGET_DIR, TRAINER_DICT


def set_config(model, prefix, code):
    protein_path = os.path.join(TARGET_DIR, f"{code}.pdb")
    protein_center = TARGET_CENTER[code]

    if model == "frag":
        from gflownet.tasks.unidock_moo_frag import moo_config

        opt_qed = "-qed" in prefix
        opt_sa = "-sa" in prefix
        config = moo_config(protein_path, protein_center, opt_qed, opt_sa)

    elif model in ["rxnflow", "synflownet", "rgfn"]:
        from gflownet.tasks.unidock_moo_synthesis import moo_config

        env_dir = sys.argv[4]
        config = moo_config(env_dir, protein_path, protein_center)

        if model == "rxnflow":
            config.algo.action_sampling.num_mc_sampling = 1
            config.algo.action_sampling.sampling_ratio_reactbi = 0.01
            config.algo.action_sampling.num_sampling_add_first_reactant = 12_000
            config.algo.action_sampling.max_sampling_reactbi = 12_000
    else:
        raise ValueError
    return config


def main():
    model = sys.argv[1]
    prefix = sys.argv[2]
    storage = sys.argv[3]

    wandb.init(group=prefix)
    code = wandb.config["protein"]
    trial = wandb.config["trial"]
    config = set_config(model, prefix, code)
    config.log_dir = os.path.join(storage, f"trial-{trial}", prefix, code)

    # NOTE: Run
    Trainer = TRAINER_DICT[model]
    trainer = Trainer(config)
    wandb.config.update({"model": model, "config": OmegaConf.to_container(trainer.cfg), "prefix": prefix})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
