from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from pmnet_appl import get_docking_proxy, BaseProxy

from gflownet import ObjectProperties

from rxnflow.config import Config
from rxnflow.base import RxnFlowSampler
from rxnflow.appl.pocket_conditional.model import RxnFlow_SinglePocket
from rxnflow.appl.pocket_conditional.utils import PocketDB
from rxnflow.appl.pocket_conditional.reward_function import get_reward_function
from rxnflow.appl.pocket_conditional.trainer import (
    PocketConditionalTask,
    PocketConditionalTrainer_SinglePocket,
    PocketConditionalTrainer_MultiPocket,
)

"""
Summary
- ProxyTask: Base Class
- ProxyTask_MultiPocket & ProxyTrainer_MultiPocket: Train Pocket-Conditioned RxnFlow.
- ProxyTask_SinglePocket & ProxyTrainer_SinglePocket: Train Pocket-Conditioned RxnFlow on single target (Few-shot).
"""


class ProxyTask(PocketConditionalTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.proxy: BaseProxy = self.models["proxy"]
        self.reward_function = get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, None, self.cfg.device)
        return {"proxy": proxy}

    def update_proxy(self):
        """add proxy cache for reward calculation"""
        cache = self.proxy.get_cache(self.protein_path, center=self.center)
        self.proxy.put_cache(self.protein_key, cache)


# TODO: check the task works well
class ProxyTask_SinglePocket(ProxyTask):
    """Single Pocket Opt (Few-shot training & sampling)"""

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.proxy: BaseProxy = self.models["proxy"]
        self.reward_function = get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging

        self.update_proxy()
        del self.proxy.pmnet  # remove unused file

    def setup_pocket_db(self):
        self.set_protein(self.cfg.task.docking.protein_path, self.cfg.task.docking.center)

    def compute_obj_properties(self, objs: list[RDMol], sample_idcs: list[int]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        _, info = self.reward_function(objs, self.protein_key)
        self.last_reward.update(info)
        flat_rewards = torch.stack([info[obj] for obj in self.objectives], dim=-1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t


class ProxyTask_MultiPocket(ProxyTask):
    """For multi-pocket environments (Pre-training)"""

    task: ProxyTask

    def setup_pocket_db(self):
        cfg = self.cfg.task.pocket_conditional
        self.pocket_db = PocketDB(torch.load(cfg.pocket_db, map_location="cpu"))

    def compute_obj_properties(self, objs: list[RDMol], sample_idcs: list[int]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in sample_idcs]
        r, info = self.reward_function(objs, pocket_keys)
        self.last_reward.update(info)
        flat_rewards = r.view(-1, 1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, "train", self.cfg.device)
        return {"proxy": proxy}


class ProxyTrainer_SinglePocket(PocketConditionalTrainer_SinglePocket):
    task: ProxyTask_SinglePocket

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED optimization for a single target"
        base.validate_every = 0
        base.task.moo.objectives = ["vina", "qed"]
        base.num_training_steps = 40_000
        base.algo.train_random_action_prob = 0.1

    def setup_task(self):
        self.task = ProxyTask_SinglePocket_Fewshot(cfg=self.cfg, wrap_model=self._wrap_for_mp)

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
        base.algo.train_random_action_prob = 0.1
        base.model.num_emb_block = 64  # TODO: train model on large GPU!

        base.cond.temperature.dist_params = [0, 64]

    def setup_task(self):
        self.task = ProxyTask_MultiPocket(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)


class ProxySampler(RxnFlowSampler):
    model: RxnFlow_SinglePocket
    task: ProxyTask_SinglePocket

    def setup_model(self):
        self.model = RxnFlow_SinglePocket(self.ctx, self.cfg, num_graph_out=self.cfg.algo.tb.do_predict_n + 1)

    def setup_task(self):
        self.task = ProxyTask_SinglePocket(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def calc_reward(self, samples: list[Any]) -> list[Any]:
        samples = super().calc_reward(samples)
        for idx, sample in enumerate(samples):
            for obj in self.task.objectives:
                sample["info"][f"reward_{obj}"] = self.task.last_reward[obj][idx]
        return samples

    @torch.no_grad()
    def set_pocket(self, protein_path: str | Path, center: tuple[float, float, float]):
        self.sampling_model.clear_cache()
        self.model.clear_cache()
        self.task.set_protein(str(protein_path), center)

    @torch.no_grad()
    def sample_against_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float],
        n: int,
        calc_reward: bool = False,
    ) -> list[dict[str, Any]]:
        """
        samples = sampler.sample_against_pocket(<pocket_file>, <center>, <n>, calc_reward = False)
        samples[0] = {'smiles': <smiles>, 'traj': <traj>, 'info': <info>}
        samples[0]['traj'] = [
            (('StartingBlock',), smiles1),        # None    -> smiles1
            (('UniMolecularReaction', template), smiles2),  # smiles1 -> smiles2
            ...                                 # smiles2 -> ...
        ]
        samples[0]['info'] = {'beta': <beta>, ...}


        samples = sampler.sample_against_pocket(..., calc_reward = True)
        samples[0]['info'] = {
            'beta': <beta>,
            'reward': <reward>,
            'reward_qed': <qed>,
            'reward_docking': <proxy>,
        }
        """
        self.set_pocket(protein_path, center)
        return self.sample(n, calc_reward)
