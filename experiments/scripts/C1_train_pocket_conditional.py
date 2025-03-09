from omegaconf import OmegaConf

import wandb
from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxyTrainer_MultiPocket

if __name__ == "__main__":

    config = init_empty(Config())
    config.env_dir = "./data/envs/catalog"
    config.task.pocket_conditional.pocket_db = "./data/experiments/CrossDocked2020/train_db.pt"
    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "CrossDocked2020")

    config.log_dir = "./logs/pocket_conditional_qvina_crossdocked2020"
    config.print_every = 10
    config.checkpoint_every = 1_000
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 8
    config.overwrite_existing_exp = True

    # change these parameters to manage memory-consumption
    config.algo.num_from_policy = 64
    config.algo.action_subsampling.sampling_ratio = 0.02

    trainer = ProxyTrainer_MultiPocket(config)

    wandb.init()
    wandb.config.update({"config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()
