from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from pmnet_appl import get_docking_proxy, BaseProxy

from gflownet import ObjectProperties
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.data.data_source import DataSource
from gflownet.utils.misc import get_worker_rng

from rxnflow.config import Config
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.appl.pocket_conditional.model import RxnFlow_SP, RxnFlow_MP
from rxnflow.appl.pocket_conditional.algo import SynthesisTB_MP
from rxnflow.appl.pocket_conditional.utils import PocketDB
from rxnflow.appl.pocket_conditional.pocket.data import generate_protein_data
from rxnflow.appl.pocket_conditional.reward_function import get_reward_function

"""
Summary
- ProxyTask: Base Class
- ProxyTask_MP & RxnFlowTrainer_MP: Train Pocket-Conditioned RxnFlow. (w/ a custom TB algorithm)
- ProxyTask_SP & RxnFlowTrainer_SP: Train Pocket-Conditioned RxnFlow on single target (Few-shot).
"""


class ProxyTask(BaseTask):
    """Sets up a task where the reward is computed using a Proxy, QED."""

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.objectives = cfg.task.moo.objectives
        self.pocket_db: PocketDB
        self.setup_pocket_db()
        self.proxy: BaseProxy = self.models["proxy"]
        self.reward_function = get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging

    @property
    def num_pockets(self) -> int:
        return len(self.pocket_db)

    def setup_pocket_db(self):
        raise NotImplementedError

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        self.proxy_model = proxy_model
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, None, self.cfg.device)
        return {"proxy": proxy}


class ProxyTask_MP(ProxyTask):
    """For multi-pocket environments"""

    def setup_pocket_db(self):
        cfg = self.cfg.task.pocket_conditional
        self.pocket_db = PocketDB(torch.load(cfg.pocket_db, map_location="cpu"))

    def compute_obj_properties(self, objs: list[RDMol], batch_idcs: list[int]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in batch_idcs]
        r, info = self.reward_function(objs, pocket_keys)
        self.last_reward.update(info)
        flat_rewards = r.view(-1, 1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        rng = get_worker_rng()
        pocket_indices = rng.choice(len(self.pocket_db), n).tolist()
        self.pocket_db.set_batch_idcs(pocket_indices)
        cond_info = super().sample_conditional_information(n, train_it)
        cond_info["pocket_global_idx"] = torch.LongTensor(pocket_indices)
        return cond_info

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.pocket_conditional.proxy
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, "train", self.cfg.device)
        return {"proxy": proxy}


class RxnFlowTrainer_MP(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED optimization for multiple targets"
        base.task.moo.objectives = ["docking", "qed"]
        base.validate_every = 0
        base.num_training_steps = 50_000
        base.algo.train_random_action_prob = 0.1
        base.model.num_emb_building_block = 64  # TODO: train model on large GPU!

        base.cond.temperature.dist_params = [16, 64]  # Different to Paper!

    def setup_task(self):
        self.task: ProxyTask_MP = ProxyTask_MP(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB_MP(self.env, self.ctx, self.cfg)

    def setup_model(self):
        self.model: RxnFlow_MP = RxnFlow_MP(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)

    def step(self, loss: Tensor):
        self.model.clear_cache()
        self.sampling_model.clear_cache()
        return super().step(loss)

    def create_data_source(self, replay_buffer: ReplayBuffer | None = None, is_algo_eval: bool = False):
        return _DataSource_MP(self.cfg, self.ctx, self.algo, self.task, replay_buffer, is_algo_eval)


class _DataSource_MP(DataSource):
    task: ProxyTask_MP

    def compute_properties(self, trajs, mark_as_online=False):
        """Sets trajs' obj_props and is_valid keys by querying the task."""
        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        objs = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]

        obj_props, m_is_valid = self.task.compute_obj_properties(
            objs, valid_idcs.tolist()
        )  # NOTE: This is only the different part
        assert obj_props.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
        # The task may decide some of the objs are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        all_fr = torch.zeros((len(trajs), obj_props.shape[1]))
        all_fr[valid_idcs] = obj_props
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = all_fr[i]
            trajs[i]["is_online"] = mark_as_online
        # Override the is_valid key in case the task made some objs invalid
        for i in valid_idcs:
            trajs[i]["is_valid"] = True


class ProxyTask_SP(ProxyTask):
    """Single Target Opt (Zero-shot sampling, Few-shot training)"""

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.models["proxy"].setup_pmnet()

    def setup_pocket_db(self):
        opt_cfg = self.cfg.task.docking
        if not OmegaConf.is_missing(opt_cfg, "protein_path"):
            self.set_protein(opt_cfg.protein_path, opt_cfg.center)

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        self._update_proxy()
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        _, info = self.reward_function(objs, self.protein_key)
        self.last_reward.update(info)
        flat_rewards = torch.stack([info[obj] for obj in self.objectives], dim=-1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t

    def set_protein(self, protein_path: str, center: tuple[float, float, float]):
        self.protein_path: str = protein_path
        self.protein_key: str = Path(self.protein_path).stem
        self.center: tuple[float, float, float] = center
        self.pocket_db = PocketDB({self.protein_key: generate_protein_data(self.protein_path, self.center)})
        self.pocket_db.set_batch_idcs([0])
        self.require_update: bool = True  # NOTE: lazy update

    def _update_proxy(self):
        if self.require_update:
            cache = self.proxy.get_cache(self.protein_path, center=self.center)
            self.proxy.put_cache(self.protein_key, cache)
            self.require_update = False


class RxnFlowTrainer_SP(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED optimization for a single target"
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.num_training_steps = 10_000

    def setup_task(self):
        self.task: ProxyTask_SP = ProxyTask_SP(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def setup_model(self):
        self.model = RxnFlow_SP(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            freeze_pocket_embedding=True,
            freeze_action_embedding=True,
        )

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.last_reward[obj].mean().item()
        super().log(info, index, key)
