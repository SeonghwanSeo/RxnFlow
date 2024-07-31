import requests
import math
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader

from rdkit.Chem import Mol as RDMol
from typing import Callable, Dict, List, Tuple

from gflownet.config import Config, init_empty
from gflownet.trainer import FlatRewards, RewardScalar

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

        self.num_cond_dim = self.temperature_conditional.encoding_size() + 2048 + 5
        with open("/home/shwan/GFLOWNET_PROJECT/astb/data/experiments/projection/train_data_05.csv") as f:
            self.train_data: List[str] = [ln.split(",")[1] for ln in f.readlines()]
        self.objectives = ["sim", "mw", "natoms", "nrings", "logp", "tpsa"]

    # def request_test_set(self):
    #     return requests.get(
    #         "https://github.com/luost26/ChemProjector/blob/main/data/synthesis_planning/test_chembl.csv",
    #         stream=True,
    #         timeout=30,
    #     )

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it, final)
        self.cond = []
        cond_features = []
        while len(self.cond) < n:
            smiles = self.train_data[np.random.choice(len(self.train_data))]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                prop, prop_t = self.get_property(mol)
                self.cond.append(prop)
                cond_features.append(prop_t)
            except Exception as e:
                raise e
                pass
        cond_features_t = torch.stack(cond_features)
        cond_info["encoding"] = torch.cat([cond_info["encoding"], cond_features_t], dim=-1)
        return cond_info

    def get_property(self, mol: RDMol):
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        natoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        nrings = rdMolDescriptors.CalcNumRings(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        fp = Chem.RDKFingerprint(mol)

        prop_t = torch.tensor([mw / 600, natoms / 10, nrings, logp, tpsa])
        fp_t = torch.tensor(fp, dtype=torch.float)
        return (
            {"mw": mw, "natoms": natoms, "nrings": nrings, "logp": logp, "tpsa": tpsa, "fp": fp},
            torch.cat([prop_t, fp_t]),
        )

    def compute_flat_rewards(self, mols: List[RDMol], indices: List[int]) -> Tuple[FlatRewards, Tensor]:
        rewards = []
        valid = []
        for mol, idx in zip(mols, indices, strict=True):
            cond = self.cond[idx]
            try:
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                natoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
                nrings = rdMolDescriptors.CalcNumRings(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                fp = Chem.RDKFingerprint(mol)

                r_sim = DataStructs.TanimotoSimilarity(fp, cond["fp"])
                r_mw = math.exp(-abs(mw - cond["mw"]) / cond["mw"])
                r_natoms = math.exp(-abs(natoms - cond["natoms"]) / cond["natoms"])
                r_nrings = math.exp(-abs(nrings - cond["nrings"]))
                r_logp = math.exp(-abs(logp - cond["logp"]) / 5)
                r_tpsa = math.exp(-abs(tpsa - cond["tpsa"]) / 100)
            except Exception as e:
                valid.append(False)
            else:
                rewards.append((r_sim, r_mw, r_natoms, r_nrings, r_logp, r_tpsa))
                valid.append(True)
        is_valid = torch.tensor(valid)
        flat_r = torch.tensor(rewards)
        self.avg_reward_info = [(obj, v) for obj, v in zip(self.objectives, flat_r.mean(0))]
        return FlatRewards(flat_r), is_valid

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """TacoGFN Reward Function: R = Prod(Rs)"""
        flat_reward = FlatRewards(torch.prod(flat_reward.clip(1e-4, 1.0), -1, keepdim=True))
        return super()._to_reward_scalar(cond_info, flat_reward)


class ProjectionSynthesisTrainer(SynthesisTrainer):
    task: ProjectionSynthesisTask

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        # cfg.cond.temperature.sample_dist = "constant"
        # cfg.cond.temperature.dist_params = [10.0]
        cfg.algo.offline_ratio = 0.25

    def setup_task(self):
        self.task = ProjectionSynthesisTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

    def setup_data(self):
        self.training_data = [None] * int(np.ceil(self.cfg.algo.global_batch_size * self.cfg.algo.offline_ratio))
        self.test_data = []

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        super().log(info, index, key)

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
    config.validate_every = 0
    config.num_training_steps = 20_000
    config.log_dir = "./logs/debug/"
    config.env_dir = "/home/shwan/GFLOWNET_PROJECT/astb/data/envs/subsampled_10000/"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True

    config.algo.action_sampling.num_mc_sampling = 1
    config.algo.action_sampling.num_sampling_add_first_reactant = 10000
    config.algo.action_sampling.ratio_sampling_reactbi = 0.1
    config.algo.action_sampling.max_sampling_reactbi = 1000
    config.algo.action_sampling.min_sampling_reactbi = 1

    trial = ProjectionSynthesisTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
