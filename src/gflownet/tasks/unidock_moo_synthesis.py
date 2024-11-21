from pathlib import Path
import numpy as np

from gflownet.config import Config, init_empty
from gflownet.misc.chem_metrics import calc_diversity

from gflownet.base.base_trainer import SynthesisTrainer, moo_trainer
from gflownet.tasks.unidock_task import UniDockMOOTask


@moo_trainer
class UniDockMOOSynthesisTrainer(SynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.validate_every = 0
        cfg.task.moo.objectives = ["vina", "qed"]

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


def moo_config(env_dir: str | Path, protein_path: str | Path, center: tuple[float, float, float]) -> Config:
    config = init_empty(Config())
    config.desc = "Vina-QED optimization with UniDock"
    config.env_dir = str(env_dir)
    config.task.docking.protein_path = str(protein_path)
    config.task.docking.center = center
    config.print_every = 1
    config.num_training_steps = 1000
    return config


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-unidock-moo-syn/"
    config.env_dir = "./data/envs/subsampled_1k/"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 1000
    config.algo.action_sampling.sampling_ratio_reactbi = 0.1
    config.algo.action_sampling.max_sampling_reactbi = 100
    config.algo.action_sampling.min_sampling_reactbi = 10

    config.task.docking.protein_path = "./data/experiments/lit-pcba-opt/protein/ADRB2_4ldo_protein.pdb"
    config.task.docking.center = (-1.96, -12.27, -48.98)

    trial = UniDockMOOSynthesisTrainer(config)
    trial.run()
