from pathlib import Path
import numpy as np
import torch

from gflownet.config import Config, init_empty

from gflownet.base.base_trainer import FragmentTrainer, MOOTrainer
from gflownet.tasks.unidock_task import UniDockMOOTask
from gflownet.tasks.unidock_moo_synthesis import calc_diversity


class UniDockMOOFragTrainer(MOOTrainer, FragmentTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.task.moo.objectives = ["vina", "qed", "sa"]

    def setup_task(self):
        self.task: UniDockMOOTask = UniDockMOOTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        if len(self.task.best_molecules) > 0:
            info["top100_n"] = len(self.task.best_molecules)
            info["top100_vina"] = np.mean([score for score, _ in self.task.best_molecules])
            if len(self.task.best_molecules) > 1:
                info["top100_div"] = calc_diversity([smi for _, smi in self.task.best_molecules])
        super().log(info, index, key)


def moo_config(
    protein_path: str | Path,
    center: tuple[float, float, float],
    qed: bool = True,
    sa: bool = True,
) -> Config:
    config = init_empty(Config())
    config.task.docking.protein_path = str(protein_path)
    config.task.docking.center = center
    config.print_every = 1
    config.num_training_steps = 1000

    if qed and (not sa):
        config.desc = "Vina-QED optimization"
        config.task.moo.objectives = ["vina", "qed"]
    elif sa and (not qed):
        config.desc = "Vina-SA optimization"
        config.task.moo.objectives = ["vina", "sa"]
    else:
        config.desc = "Vina-QED-SA optimization"
        config.task.moo.objectives = ["vina", "qed", "sa"]
    return config


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 0
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-unidock-moo-frag/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./data/experiments/lit-pcba-opt/protein/ADRB2_4ldo_protein.pdb"
    config.task.docking.center = (-1.96, -12.27, -48.98)
    trial = UniDockMOOFragTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
