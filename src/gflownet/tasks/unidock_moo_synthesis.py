from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Union
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config, init_empty
from gflownet.trainer import FlatRewards, RewardScalar
from gflownet.tasks.unidock_synthesis import UniDockTask
from gflownet.tasks.synthesis_trainer import SynthesisTrainer

from rdkit.Chem import QED
from gflownet.utils import sascore


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def mol2sas(mols: list[RDMol], is_valid: list[bool], default=10):
    sas = torch.tensor([safe(sascore.calculateScore, i, default) for i, v in zip(mols, is_valid) if v])
    sas = (10 - sas) / 9  # Turn into a [0-1] reward
    return sas


def mol2qed(mols: list[RDMol], is_valid: list[bool], default=0):
    return torch.tensor([safe(QED.qed, i, default) for i, v in zip(mols, is_valid) if v])


aux_tasks = {"qed": mol2qed, "sa": mol2sas}


class UniDockMOOTask(UniDockTask):
    """Sets up a task where the reward is computed using a UniDock, QED, SAScore - TacoGFN Task."""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module],
    ):
        super().__init__(cfg, rng, wrap_model)
        mcfg = self.cfg.task.unidock_moo
        self.objectives: List[str] = mcfg.objectives
        self.threshold: Dict[str, float] = {
            "vina": mcfg.vina_threshold,
            "qed": mcfg.qed_threshold,
            "sa": mcfg.sa_score_threshold,
        }
        self.weight: Dict[str, float] = {
            "vina": mcfg.vina_weight,
            "qed": mcfg.qed_weight,
            "sa": mcfg.sa_score_weight,
        }
        assert set(self.objectives) <= {"vina", "qed", "sa"} and len(self.objectives) == len(set(self.objectives))
        self.avg_reward_info = []

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        is_valid = [True] * len(mols)
        is_valid_t = torch.tensor(is_valid)

        flat_r: List[Tensor] = []
        for obj in self.objectives:
            if obj == "vina":
                flat_r.append(super().compute_unidock_rewards(mols))
            else:
                flat_r.append(aux_tasks[obj](mols, is_valid))
        flat_rewards = torch.stack(flat_r, dim=1)
        self.avg_reward_info = [(obj, v) for obj, v in zip(self.objectives, flat_rewards.mean(0))]
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """TacoGFN Reward Function: R = Prod(Rs)"""
        weighted_reward = self.weighting_reward(flat_reward)
        flat_reward = FlatRewards(torch.prod(weighted_reward, -1, keepdim=True).clip(1e-4, 1.0))
        return super()._to_reward_scalar(cond_info, flat_reward)

    def weighting_reward(self, score: torch.Tensor) -> torch.Tensor:
        threshold = torch.tensor([self.threshold[obj] for obj in self.objectives], dtype=torch.float)
        weight = torch.tensor([self.weight[obj] for obj in self.objectives], dtype=torch.float)
        return (score / threshold).clip(0.0, 1.0).pow(weight)


class UniDockMOOSynthesisTrainer(SynthesisTrainer):
    task: UniDockMOOTask

    def set_default_hps(self, cfg):
        super().set_default_hps(cfg)
        cfg.task.unidock_moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = UniDockMOOTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        if len(self.task.best_molecules) > 0:
            info["sampled_vina_top100"] = np.mean([score for score, _ in self.task.best_molecules])
        super().log(info, index, key)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        state = torch.load(checkpoint_path, map_location="cpu")
        print(f"load pre-trained model from {checkpoint_path}")
        self.model.load_state_dict(state["models_state_dict"][0])
        if self.sampling_model is not self.model:
            self.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 0
    config.num_training_steps = 1000
    config.log_dir = "./logs/debug/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/subsampled_10000/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 1000
    config.algo.action_sampling.ratio_sampling_reactbi = 0.1
    config.algo.action_sampling.max_sampling_reactbi = 1000

    config.task.unidock.code = "14gs_A"

    trial = UniDockMOOSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
