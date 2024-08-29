from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor
from omegaconf import OmegaConf

from pmnet_appl.largfn_reward import get_unidock_proxy
from pmnet_appl.tacogfn_reward import get_qvina_proxy

from gflownet.config import Config
from gflownet.trainer import FlatRewards
from gflownet.base.base_task import BaseTask
from gflownet.sbdd.utils import PocketDB
from gflownet.sbdd.pocket.data import generate_protein_data
from gflownet.misc import chem_metrics


def tacogfn_reward_function(affinity: Tensor, qed: Tensor, sa: Tensor) -> Tensor:
    """TacoGFN Reward Function
    https://github.com/tsa87/TacoGFN-SBDD/blob/main/src/tacogfn/tasks/pharmaco_frag.py
    r_aff:
        - 0                             (0 <= affinity)
        - -affinity * 0.04              (-7 <= affinity <= 0)
        - (-affinity - 7) * 0.2 + 0.28  (-12 <= affinity <= -7)
        - 1.28                          (affinity <= -12)
    r_qed:
        - qed * 0.7                     (0 <= qed <= 0.7)
        - 1.0                           (0.7 <= qed)
    r_sa:
        - sa * 0.8                      (0 <= sa <= 0.8)
        - 1.0                           (0.8 <= sa)
    r = 3 * r_aff * r_qed * r_sa
    """
    affinity = ((affinity + 7).clip(-5, 0) + 0.2 * affinity.clip(min=-7)) / -5
    qed = (qed / 0.7).clip(0, 1)
    sa = (sa / 0.8).clip(0, 1)
    return 3 * affinity * qed * sa


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
        assert set(self.objectives) <= {"docking", "qed", "sa"}
        sbdd_cfg = cfg.task.sbdd
        pocket_db_dict = torch.load(sbdd_cfg.pocket_db, map_location="cpu")
        self.pocket_db: PocketDB = PocketDB(pocket_db_dict)
        self.num_pockets: int = len(self.pocket_db)
        self.last_reward: dict[str, Tensor] = {}

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)

        if "docking" in self.objectives:
            self.last_reward["docking"] = affinity = self.mol2proxy(mols, batch_idx)
        else:
            affinity = torch.full((len(mols),), -12.0, dtype=torch.float)

        if "qed" in self.objectives:
            self.last_reward["qed"] = qed = chem_metrics.mol2qed(mols)
        else:
            qed = torch.ones((len(mols),), dtype=torch.float)

        if "sa" in self.objectives:
            self.last_reward["sa"] = sa = chem_metrics.mol2sascore(mols)
        else:
            sa = torch.ones((len(mols),), dtype=torch.float)

        flat_rewards = tacogfn_reward_function(affinity, qed, sa).view(-1, 1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        pocket_indices = self.rng.choice(self.num_pockets, n)
        self.pocket_db.set_batch_idcs(pocket_indices.tolist())
        cond_info = super().sample_conditional_information(n, train_it, final)
        cond_info["pocket_global_idx"] = torch.LongTensor(pocket_indices)
        return cond_info

    def mol2proxy(self, mols: list[RDMol], batch_idx: list[int]) -> Tensor:
        proxy_model = self.models["proxy"]
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in batch_idx]
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        vinas = [
            float(proxy_model.scoring(pocket_key, smi)) for pocket_key, smi in zip(pocket_keys, smiles, strict=True)
        ]
        return torch.tensor(vinas, dtype=torch.float32)

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_type, proxy_dataset = self.cfg.task.sbdd.proxy.split("-")
        assert proxy_type in ("qvina", "unidock"), f"proxy type({proxy_type}) should be `qvina` or `unidock`"
        assert proxy_dataset in (
            "ZINCDock15M",
            "CrossDocked2020",
        ), f"dataset({proxy_dataset}) should be `CrossDocked2020` or `ZINCDock15M`"
        if proxy_type == "qvina":
            proxy_model = get_qvina_proxy(proxy_dataset, "train", "cpu")
        else:
            assert (
                proxy_dataset == "ZINCDock15M"
            ), f"For unidock proxy, dataset({proxy_dataset}) should be `ZINCDock15M`"
            proxy_model = get_unidock_proxy("train", "cpu")
        return {"proxy": proxy_model}


class SBDD_SingleOpt_Task(SBDDTask):
    """Single Target Opt (Zero-shot, Few-shot)"""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module],
    ):
        BaseTask.__init__(self, cfg, rng, wrap_model)
        self.objectives = cfg.task.moo.objectives
        assert set(self.objectives) <= {"docking", "qed", "sa"}

        opt_cfg = cfg.task.docking
        if not OmegaConf.is_missing(opt_cfg, "protein_path"):
            self.set_protein(opt_cfg.protein_path, opt_cfg.center)
        self.last_reward: dict[str, Tensor] = {}
        self.do_proxy_update: bool = True

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

    def mol2proxy(self, mols: list[RDMol], batch_idx: list[int]) -> Tensor:
        proxy_model = self.models["proxy"]
        if self.do_proxy_update:
            cache = proxy_model.feature_extraction(self.protein_path, center=self.center)
            proxy_model.put_cache(self.protein_key, cache)
            self.protein_updated = False

        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return proxy_model.scoring_list(self.protein_key, smiles).to("cpu", torch.float32)

    def _load_task_models(self) -> dict[str, nn.Module]:
        proxy_type, proxy_dataset = self.cfg.task.sbdd.proxy.split("-")
        assert proxy_type in ("qvina", "unidock"), f"proxy type({proxy_type}) should be `qvina` or `unidock`"
        assert proxy_dataset in (
            "ZINCDock15M",
            "CrossDocked2020",
        ), f"dataset({proxy_dataset}) should be `CrossDocked2020` or `ZINCDock15M`"
        if proxy_type == "qvina":
            proxy_model = get_qvina_proxy(proxy_dataset, None, "cpu")
        else:
            assert (
                proxy_dataset == "ZINCDock15M"
            ), f"For unidock proxy, dataset({proxy_dataset}) should be `ZINCDock15M`"
            proxy_model = get_unidock_proxy(None, "cpu")
        proxy_model.setup_pmnet()
        return {"proxy": proxy_model}
