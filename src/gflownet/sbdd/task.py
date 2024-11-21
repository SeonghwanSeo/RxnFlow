from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor
from omegaconf import OmegaConf

from pmnet_appl import get_docking_proxy, BaseProxy

from gflownet.config import Config
from gflownet.trainer import FlatRewards
from gflownet.base.base_task import BaseTask
from gflownet.sbdd.utils import PocketDB
from gflownet.sbdd.pocket.data import generate_protein_data
from gflownet.sbdd.reward_function import get_reward_function


class SBDDTask(BaseTask):
    """Sets up a task where the reward is computed using a Proxy, QED."""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module],
    ):
        super().__init__(cfg, rng, wrap_model)
        self.objectives = cfg.task.moo.objectives
        self.setup_pocket_db()
        self.last_reward: dict[str, Tensor] = {}
        self.reward_function = get_reward_function(self.models["proxy"], self.objectives)

    def setup_pocket_db(self):
        sbdd_cfg = self.cfg.task.sbdd
        pocket_db_dict = torch.load(sbdd_cfg.pocket_db, map_location="cpu")
        self.pocket_db: PocketDB = PocketDB(pocket_db_dict)
        self.num_pockets: int = len(self.pocket_db)

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in batch_idx]
        r, info = self.reward_function(mols, pocket_keys)
        self.last_reward.update(info)
        flat_rewards = r.view(-1, 1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        pocket_indices = self.rng.choice(self.num_pockets, n)
        self.pocket_db.set_batch_idcs(pocket_indices.tolist())
        cond_info = super().sample_conditional_information(n, train_it, final)
        cond_info["pocket_global_idx"] = torch.LongTensor(pocket_indices)
        return cond_info

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.sbdd.proxy
        self.proxy_model = proxy_model
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, "train", self.cfg.device)
        return {"proxy": proxy}


class SBDD_SingleOpt_Task(SBDDTask):
    """Single Target Opt (Zero-shot, Few-shot)"""

    def __init__(self, cfg: Config, rng: np.random.Generator, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, rng, wrap_model)
        self.models["proxy"].setup_pmnet()

    def setup_pocket_db(self):
        opt_cfg = self.cfg.task.docking
        if not OmegaConf.is_missing(opt_cfg, "protein_path"):
            self.set_protein(opt_cfg.protein_path, opt_cfg.center)

    def set_protein(self, protein_path: str, center: tuple[float, float, float]):
        self.protein_path: str = protein_path
        self.protein_key: str = Path(self.protein_path).stem
        self.center: tuple[float, float, float] = center
        self.do_proxy_update: bool = True  # NOTE: lazy update

        pocket_db_dict = {self.protein_key: generate_protein_data(self.protein_path, self.center)}
        self.pocket_db = PocketDB(pocket_db_dict)
        self.num_pockets: int = 1
        self.pocket_db.set_batch_idcs([0])

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        return BaseTask.sample_conditional_information(self, n, train_it, final)

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        self._update_proxy()
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)
        _, info = self.reward_function(mols, self.protein_key)
        self.last_reward.update(info)
        flat_rewards = torch.stack([info[obj] for obj in self.objectives], dim=-1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def _update_proxy(self):
        proxy: BaseProxy = self.models["proxy"]
        if self.do_proxy_update:
            cache = proxy.get_cache(self.protein_path, center=self.center)
            proxy.put_cache(self.protein_key, cache)
            self.do_protein_updated = False

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_model, proxy_type, proxy_dataset = self.cfg.task.sbdd.proxy
        self.proxy_model = proxy_model
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, None, "cuda")
        return {"proxy": proxy}
