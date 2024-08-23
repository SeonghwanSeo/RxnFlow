from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd

from rdkit import Chem

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from pmnet_appl.largfn_reward import get_unidock_proxy
from pmnet_appl.tacogfn_reward import get_qvina_proxy

from gflownet.config import Config
from gflownet.trainer import FlatRewards
from gflownet.misc import chem_metrics
from gflownet.base.base_task import BaseMOOTask

from gflownet.sbdd.gvp_data import generate_protein_data


class PocketDB:
    def __init__(self, pocket_graph_dict: dict[str, dict[str, Tensor]]):
        self.keys: list[str] = list(pocket_graph_dict.keys())
        self.graphs: list[gd.Data] = [gd.Data(**pocket_graph_dict[key]) for key in self.keys]

        self.batch_idx: list[int]
        self.batch_keys: list[str]
        self.batch_g: gd.Batch

    def __len__(self):
        return len(self.keys)

    def set_batch_idcs(self, indices: list[int]):
        self.batch_idcs = indices
        self.batch_keys = [self.keys[i] for i in indices]
        self.batch_g = gd.Batch.from_data_list([self.graphs[i] for i in indices])

    def pocket_idx_to_batch_idx(self, indices: list[int]):
        return [self.batch_idcs.index(idx) for idx in indices]


class SBDDTask(BaseMOOTask):
    """Sets up a task where the reward is computed using a Proxy, QED."""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module],
    ):
        super().__init__(cfg, rng, wrap_model)
        assert set(self.objectives) <= {"docking", "qed", "sa"}

        sbdd_cfg = cfg.task.sbdd
        pocket_db_dict = torch.load(sbdd_cfg.pocket_db, map_location="cpu")
        self.pocket_db: PocketDB = PocketDB(pocket_db_dict)
        self.num_pockets: int = len(self.pocket_db)

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

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        pocket_indices = self.rng.choice(self.num_pockets, n)
        self.pocket_db.set_batch_idcs(pocket_indices.tolist())
        cond_info = super().sample_conditional_information(n, train_it, final)
        cond_info["pocket_global_idx"] = torch.LongTensor(pocket_indices)
        return cond_info

    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)

        flat_r: list[Tensor] = []
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "docking":
                fr = self.mol2proxy(mols, batch_idx)
                flat_r.append(fr * 0.1)
            elif obj == "sa":
                fr = chem_metrics.mol2qed(mols)
                flat_r.append(fr)
            else:
                fr = chem_metrics.mol2sascore(mols)
                flat_r.append(fr)
            self.avg_reward_info.append((obj, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return FlatRewards(flat_rewards), is_valid_t

    def mol2proxy(self, mols: list[RDMol], batch_idx: list[int]) -> Tensor:
        proxy_model = self.models["proxy"]
        pocket_keys = [self.pocket_db.batch_keys[idx] for idx in batch_idx]
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        vinas = [
            float(proxy_model.scoring(pocket_key, smi)) for pocket_key, smi in zip(pocket_keys, smiles, strict=True)
        ]
        return torch.tensor(vinas, dtype=torch.float32).neg().clip(min=0.0)


class SBDD_SingleOpt_Task(SBDDTask):
    """Single Target Opt (Zero-shot, Few-shot)"""

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module],
    ):
        BaseMOOTask.__init__(self, cfg, rng, wrap_model)
        assert set(self.objectives) <= {"docking", "qed", "sa"}

        opt_cfg = cfg.task.docking
        self.protein_path: str = opt_cfg.protein_path
        self.center: tuple[float, float, float] = opt_cfg.center
        self.protein_key: str = Path(self.protein_path).stem

        pocket_db_dict = {self.protein_key: generate_protein_data(self.protein_path, self.center)}
        self.pocket_db = PocketDB(pocket_db_dict)
        self.num_pockets: int = 1
        self.pocket_db.set_batch_idcs([0])

        proxy_model = self.models["proxy"]
        cache = proxy_model.feature_extraction(self.protein_path, center=self.center)
        proxy_model.put_cache(self.protein_key, cache)

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> dict[str, Tensor]:
        cond_info = BaseMOOTask.sample_conditional_information(self, n, train_it, final)
        return cond_info

    def mol2proxy(self, mols: list[RDMol], batch_idx: list[int]) -> Tensor:
        proxy_model = self.models["proxy"]
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return proxy_model.scoring_list(self.protein_key, smiles).to(torch.float32).neg().clip(min=0.0)

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
        return {"proxy": proxy_model}
