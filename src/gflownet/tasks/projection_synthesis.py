import requests
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader

from rdkit.Chem import Mol as RDMol
from typing import Callable, Dict, List, Tuple

from gflownet.config import Config, init_empty
from gflownet.trainer import FlatRewards

from gflownet.tasks.synthesis_trainer import SynthesisTrainer
from gflownet.tasks.base_task import BaseTask

from gflownet.misc.projection.sampling_iterator import ProjectionSamplingIterator


class ProjectionSynthesisTask(BaseTask):
    """Sets up a task where the reward is structural similarity to target molecule."""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        super().__init__(cfg, rng, wrap_model)

        self.num_cond_dim = self.temperature_conditional.encoding_size() + 2048
        with open("/home/shwan/GFLOWNET_PROJECT/astb/data/experiments/projection/train_data_05.csv") as f:
            self.train_data: List[str] = [ln.split(",")[1] for ln in f.readlines()]

    # def request_test_set(self):
    #     return requests.get(
    #         "https://github.com/luost26/ChemProjector/blob/main/data/synthesis_planning/test_chembl.csv",
    #         stream=True,
    #         timeout=30,
    #     )

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it, final)

        self.cond_fps = []
        while len(self.cond_fps) < n:
            smiles = self.train_data[np.random.choice(len(self.train_data))]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                self.cond_fps.append(Chem.RDKFingerprint(mol))
            except Exception:
                pass
        fp_tensor = torch.tensor(self.cond_fps)
        cond_info["encoding"] = torch.cat([cond_info["encoding"], fp_tensor], dim=-1)
        return cond_info

    def compute_flat_rewards(self, mols: List[RDMol], indices: List[int]) -> Tuple[FlatRewards, Tensor]:
        sim_list = []
        valid = []
        for mol, idx in zip(mols, indices, strict=True):
            cond_fp = self.cond_fps[idx]
            try:
                fp = Chem.RDKFingerprint(mol)
                sim = DataStructs.TanimotoSimilarity(fp, cond_fp)
            except Exception:
                valid.append(False)
            else:
                sim_list.append(sim)
                valid.append(True)
        is_valid = torch.tensor(valid)
        rewards = self.flat_reward_transform(torch.tensor(sim_list)).clip(1e-4, 1).reshape((-1, 1))
        return FlatRewards(rewards), is_valid


class ProjectionSynthesisTrainer(SynthesisTrainer):
    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [10.0]
        cfg.algo.offline_ratio = 0.25

    def setup_task(self):
        self.task = ProjectionSynthesisTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def setup_data(self):
        self.training_data = [None] * int(np.ceil(self.cfg.algo.global_batch_size * self.cfg.algo.offline_ratio))

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = ProjectionSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            det_after=self.cfg.algo.train_det_after,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=False,
            prefetch_factor=None,
        )

    def build_final_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        iterator = ProjectionSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            ratio=0.0,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            random_action_prob=0.0,
            hindsight_ratio=0.0,
            init_train_iter=self.cfg.num_training_steps,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=False,
            prefetch_factor=None,
        )


def main():
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.validate_every = 10
    config.num_training_steps = 1000
    config.log_dir = "./logs/debug/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/subsampled_20000/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 5000
    config.algo.action_sampling.ratio_sampling_reactbi = 1.0
    config.algo.action_sampling.max_sampling_reactbi = 5000
    config.algo.action_sampling.min_sampling_reactbi = 1

    trial = ProjectionSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
