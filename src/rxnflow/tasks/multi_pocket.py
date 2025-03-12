import warnings
from pathlib import Path
from typing import Any

import torch
from pmnet_appl import BaseProxy, get_docking_proxy
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from rxnflow.appl.pocket_conditional.model import RxnFlow_SinglePocket
from rxnflow.appl.pocket_conditional.pocket.data import generate_protein_data
from rxnflow.appl.pocket_conditional.reward_function import get_reward_function
from rxnflow.appl.pocket_conditional.trainer import (
    PocketConditionalTask,
    PocketConditionalTrainer_MultiPocket,
    PocketConditionalTrainer_SinglePocket,
)
from rxnflow.appl.pocket_conditional.utils import PocketDB, get_mol_center
from rxnflow.base import RxnFlowSampler
from rxnflow.config import Config

"""
Summary
- ProxyTask: Base Class
- ProxyTask_MultiPocket & ProxyTrainer_MultiPocket: Train Pocket-Conditioned RxnFlow.
- ProxyTask_SinglePocket & ProxyTrainer_SinglePocket: Train Pocket-Conditioned RxnFlow on single target (Few-shot).
"""


class ProxyTask(PocketConditionalTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.proxy: BaseProxy = self._load_task_models()
        self.reward_function = get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging

    def _load_task_models(self) -> BaseProxy:
        assert self.cfg.task.pocket_conditional.proxy, "cfg.task.pocket_conditional.proxy is required"
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, None, self.cfg.device)
        return proxy


class ProxyTask_SinglePocket(ProxyTask):
    """Single Pocket Opt (Few-shot training & sampling)"""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.reward_function = get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging

    def compute_obj_properties(
        self, mols: list[RDMol], sample_idcs: list[int] | None = None
    ) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)
        _, info = self.reward_function(mols, self.protein_key)
        self.last_reward.update(info)
        flat_rewards = torch.stack([info[obj] for obj in self.objectives], dim=-1)
        assert flat_rewards.shape[0] == len(mols)
        return ObjectProperties(flat_rewards), is_valid_t

    def set_protein(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
    ):
        """set single protein db"""
        # get center
        if center is None:
            assert ref_ligand_path is not None, "One of center or reference ligand path is required"
            center = get_mol_center(str(ref_ligand_path))
        else:
            if ref_ligand_path is not None:
                warnings.warn(
                    "Both `center` and `ref_ligand_path` are given, so the reference ligand is ignored", stacklevel=2
                )

        self.protein_path: str = str(protein_path)
        self.protein_key: str = Path(self.protein_path).stem
        self.center: tuple[float, float, float] = center
        self.pocket_db = PocketDB({self.protein_key: generate_protein_data(self.protein_path, self.center)})
        self.pocket_db.set_batch_idcs([0])

        # calculate pmnet-proxy cache
        cache = self.proxy.get_cache(self.protein_path, center=self.center)
        self.proxy.put_cache(self.protein_key, cache)


class ProxyTask_Sampling(ProxyTask_SinglePocket):
    def setup_pocket_db(self):
        pass


class ProxyTask_Fewshot(ProxyTask_SinglePocket):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        del self.proxy.pmnet

    def setup_pocket_db(self):
        assert self.cfg.task.docking.protein_path, "cfg.task.docking.protein_path is required"
        self.set_protein(
            self.cfg.task.docking.protein_path, self.cfg.task.docking.center, self.cfg.task.docking.ref_ligand_path
        )


class ProxyTask_MultiPocket(ProxyTask):
    """For multi-pocket environments (Pre-training)"""

    def setup_pocket_db(self):
        cfg = self.cfg.task.pocket_conditional
        assert cfg.pocket_db, "cfg.task.pocket_conditional.pocket_db is required"
        self.pocket_db = PocketDB(torch.load(cfg.pocket_db, map_location="cpu"))

    def compute_obj_properties(self, mols: list[RDMol], sample_idcs: list[int]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in sample_idcs]
        r, info = self.reward_function(mols, pocket_keys)
        self.last_reward.update(info)
        flat_rewards = r.view(-1, 1)
        assert flat_rewards.shape[0] == len(mols)
        return ObjectProperties(flat_rewards), is_valid_t

    def _load_task_models(self) -> BaseProxy:
        assert self.cfg.task.pocket_conditional.proxy, "cfg.task.pocket_conditional.proxy is required"
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, "train", self.cfg.device)
        return proxy


class ProxyTrainer_Fewshot(PocketConditionalTrainer_SinglePocket):
    task: ProxyTask_Fewshot

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED optimization for a single target"
        base.task.moo.objectives = ["vina", "qed"]
        base.validate_every = 0
        base.num_training_steps = 50_000
        base.algo.train_random_action_prob = 0.1

        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]

    def setup_task(self):
        self.task = ProxyTask_Fewshot(cfg=self.cfg)

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)


class ProxyTrainer_MultiPocket(PocketConditionalTrainer_MultiPocket):
    task: ProxyTask_MultiPocket

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED optimization for multiple targets"
        base.task.moo.objectives = ["vina", "qed"]
        base.validate_every = 0
        base.num_training_steps = 50_000
        base.algo.train_random_action_prob = 0.2

        # GFN parameters
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]

        # replay buffer is not supported
        base.replay.use = False

        # training learning rate
        base.opt.learning_rate = 1e-4
        base.opt.lr_decay = 10_000
        base.algo.tb.Z_learning_rate = 1e-2
        base.algo.tb.Z_lr_decay = 20_000

        # pretrain -> more train and better regularization with dropout
        base.model.dropout = 0.1

    def setup_task(self):
        self.task = ProxyTask_MultiPocket(cfg=self.cfg)

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)


class ProxySampler(RxnFlowSampler):
    model: RxnFlow_SinglePocket
    task: ProxyTask_Sampling

    @torch.no_grad()
    def set_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
    ):
        """Change pocket"""
        self.model.clear_cache()
        self.task.set_protein(protein_path, center, ref_ligand_path)

    @torch.no_grad()
    def sample_against_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float],
        n: int,
        calc_reward: bool = False,
    ) -> list[dict[str, Any]]:
        """
        # generation only
        samples = sampler.sample_against_pocket(<pocket_file>, <center>, <n>, calc_reward = False)
        samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
        samples[0]['traj'] = [
            (('Firstblock', block), smiles1),       # None    -> smiles1
            (('UniRxn', template), smiles2),        # smiles1 -> smiles2
            (('BiRxn', template, block), smiles3),  # smiles2 -> smiles3
            ...                                     # smiles3 -> ...
        ]
        samples[0]['info'] = {'beta': <beta>, ...}

        # with rewarding
        samples = sampler.sample_against_pocket(..., calc_reward = True)
        samples[0]['info'] = {
            'beta': <beta>,
            'reward': <reward>,
            'reward_qed': <qed>,
            'reward_vina': <proxy>,
        }
        """
        self.set_pocket(protein_path, center)
        return self.sample(n, calc_reward)

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(self.ctx, self.cfg, num_graph_out=self.cfg.algo.tb.do_predict_n + 1)

    def setup_task(self):
        self.task = ProxyTask_Sampling(cfg=self.cfg)

    def calc_reward(self, samples: list[Any]) -> list[Any]:
        samples = super().calc_reward(samples)
        for idx, sample in enumerate(samples):
            for obj in self.task.objectives:
                sample["info"][f"reward_{obj}"] = self.task.last_reward[obj][idx]
        return samples
